"""Converts raw audio signal to a dataset of TFRecords for training."""
import multiprocessing
import os
from copy import deepcopy

import numpy as np
import tensorflow as tf
from absl import app, flags
from loguru import logger
from tqdm import tqdm

from vad.data_processing.data_iterator import slice_iter, split_data
from vad.data_processing.feature_extraction import extract_features

flags.DEFINE_string(
    "data_dir", "/home/filippo/datasets/LibriSpeech/", "data directory path"
)
flags.DEFINE_string("data_split", "0.7/0.15", "train/val split")
flags.DEFINE_integer("seq_len", 1024, "segment length")
flags.DEFINE_integer("num_shards", 256, "number of tfrecord files")
flags.DEFINE_boolean("debug", False, "debug for a few samples")
flags.DEFINE_string("data_type", "trainvaltest", "data types to write into tfrecords")

FLAGS = flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


def int64_feature(value):
    """Utilitary function to build a TF feature from an int.

    Args:
        value ([int]): value to convert to a TF feature

    Returns:
        (tf.core.example.feature_pb2.Feature): int TF feature
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    """Utilitary function to build a TF feature from a list of ints.

    Args:
        value (list): value to convert to a TF feature

    Returns:
        (tf.core.example.feature_pb2.Feature): list of ints TF feature
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    """Utilitary function to build a TF feature from a byte.

    Args:
        value (byte): value to convert to a TF feature

    Returns:
        (tf.core.example.feature_pb2.Feature): byte TF feature
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    """Utilitary function to build a TF feature from a list of bytes.

    Args:
        value (list): value to convert to a TF feature

    Returns:
        (tf.core.example.feature_pb2.Feature): list of bytes TF feature
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    """Utilitary function to build a TF feature from a float.

    Args:
        value (list): value to convert to a TF feature

    Returns:
        (tf.core.example.feature_pb2.Feature): float TF feature
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    """Utilitary function to build a TF feature from a list of floats.

    Args:
        value (list): value to convert to a TF feature

    Returns:
        (tf.core.example.feature_pb2.Feature): list of floats TF feature
    """
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tf_example(
    file_id,
    start,
    end,
    sub_segment,
    sub_segment_id,
    sub_segment_len,
    label,
):
    """Build a TF training example from raw data.

    Args:
        file_id (int): file ID
        start (int): global audio segment start frame
        end (int): global audio segment end frame
        sub_segment (np.ndarray): sub audio segment signal
        sub_segment_id (int): sub audio segment ID
        sub_segment_len (int): sub audio segment length
        label (int): sub audio segment ID (0 or 1)

    Returns:
        example (tf.core.example.example_pb2.Example): TF training example
    """
    # Drop 1/4th of speech signals to balance data
    if label == 1:
        drop = np.random.randint(0, 4)
        if drop > 0:
            return None

    # MFCC feature extraction
    signal_to_process = np.copy(sub_segment)
    signal_to_process = np.float32(signal_to_process)
    features = extract_features(
        signal_to_process, freq=16000, n_mfcc=5, size=512, step=16
    )
    features = np.reshape(features, -1)

    feature_dict = {
        "signal/id": bytes_feature(file_id.encode()),
        "segment/start": int64_feature(int(start)),
        "segment/end": int64_feature(int(end)),
        "subsegment/id": int64_feature(sub_segment_id),
        "subsegment/length": int64_feature(sub_segment_len),
        "subsegment/signal": float_list_feature(sub_segment.tolist()),
        "subsegment/features": float_list_feature(features.tolist()),
        "subsegment/label": int64_feature(label),
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def pool_create_tf_example(args):
    """Utilitary function to use multi-processing to create TF examples.

    Args:
        args (tuple): create_tf_example function args

    Returns:
        (tf.core.example.example_pb2.Example): TF training example
    """
    return create_tf_example(*args)


def write_tfrecords(path, dataiter, num_shards=256, nmax=-1):
    """Write a TFRecords dataset.

    Args:
        path (str): path to output directory
        dataiter (generate): data iterator
        num_shards (int, optional): number of TFRecords to create. Defaults to 256.
        nmax (int, optional): max number of data elements to create. Defaults to -1 for all data.
    """
    writers = [
        tf.python_io.TFRecordWriter(f"{path}{i:05d}_{num_shards:05d}.tfrecord")
        for i in range(num_shards)
    ]
    logger.info(f"Writing to output path: {path}")
    pool = multiprocessing.Pool()
    counter = 0
    for i, tf_example in tqdm(
        enumerate(
            pool.imap(
                pool_create_tf_example,
                [
                    (
                        deepcopy(data["file_id"]),
                        deepcopy(data["start"]),
                        deepcopy(data["end"]),
                        deepcopy(data["sub_segment"]),
                        deepcopy(data["sub_segment_id"]),
                        deepcopy(data["sub_segment_len"]),
                        deepcopy(data["label"]),
                    )
                    for data in dataiter
                ],
            )
        )
    ):
        if tf_example is not None:
            writers[i % num_shards].write(tf_example.SerializeToString())
            counter += 1
        if 0 < nmax < i:
            break
    pool.close()
    for writer in writers:
        writer.close()
    logger.info(f"Recorded {counter} signals")


def create_tfrecords(
    data_dir,
    seq_len=1024,
    split="0.7/0.15",
    num_shards=256,
    debug=False,
    data_type="trainval",
):
    """Create a dataset of TFRecords for VAD.

    Args:
        data_dir (str): path to data directory
        seq_len (int, optional): sub segment window length. Defaults to 1024.
        split (str, optional): train / val split. Defaults to "0.7/0.15".
        num_shards (int, optional): number of TFRecords to create. Defaults to 256.
        debug (bool, optional): debug with a small amount of data. Defaults to False.
        data_type (str, optional): dataset subsets to create. Defaults to "trainval".
    """
    np.random.seed(0)

    output_path = os.path.join(data_dir, "tfrecords/")
    if not tf.gfile.IsDirectory(output_path):
        tf.gfile.MakeDirs(output_path)

    # Data & label directories
    label_dir = os.path.join(data_dir, "labels/")
    data_dir = os.path.join(data_dir, "test-clean/")

    # Split data on files
    train, val, test = split_data(label_dir, split, random_seed=0)

    tot_files = len(train) + len(val) + len(test)
    logger.info(f"Total files: {tot_files}")
    logger.info(f"Train/val/test split: {len(train)}/{len(val)}/{len(test)}")
    train_it = slice_iter(data_dir, label_dir, train, seq_len)
    val_it = slice_iter(data_dir, label_dir, val, seq_len)
    test_it = slice_iter(data_dir, label_dir, test, seq_len)

    # Write data to tfrecords format
    nmax = 100 if debug else -1
    if "train" in data_type:
        logger.info("Writing train tfrecords ...")
        train_path = os.path.join(output_path, "train/")
        if not tf.gfile.IsDirectory(train_path):
            tf.gfile.MakeDirs(train_path)
        write_tfrecords(train_path, train_it, num_shards, nmax=nmax)

    if "val" in data_type:
        logger.info("Writing val tfrecords ...")
        val_path = os.path.join(output_path, "val/")
        if not tf.gfile.IsDirectory(val_path):
            tf.gfile.MakeDirs(val_path)
        write_tfrecords(val_path, val_it, num_shards, nmax=nmax)

    if "test" in data_type:
        logger.info("Writing test tfrecords ...")
        test_path = os.path.join(output_path, "test/")
        if not tf.gfile.IsDirectory(test_path):
            tf.gfile.MakeDirs(test_path)
        write_tfrecords(test_path, test_it, num_shards, nmax=nmax)


def main(_):
    """Main function to create a TFRecords dataset for VAD.

    Args:
        _ ([type]): [description]
    """
    create_tfrecords(
        FLAGS.data_dir,
        seq_len=FLAGS.seq_len,
        split=FLAGS.data_split,
        num_shards=FLAGS.num_shards,
        debug=FLAGS.debug,
        data_type=FLAGS.data_type,
    )


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)
