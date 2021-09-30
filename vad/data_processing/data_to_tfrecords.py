"""Converts raw audio signal to a dataset of TFRecords for training."""
import argparse
import multiprocessing
import os
from copy import deepcopy

import numpy as np
import tensorflow as tf
from loguru import logger
from tqdm import tqdm

from vad.data_processing.data_iterator import slice_iter, split_data
from vad.data_processing.feature_extraction import extract_features


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


def create_tfrecords(params, data_dir, debug=False):
    """Create a dataset of TFRecords for VAD.

    Args:
        params (dict): dataset parameters
        data_dir (str): path to data directory
        debug (bool, optional): debug with a small amount of data. Defaults to False.
    """
    tf.logging.set_verbosity(tf.logging.INFO)
    np.random.seed(0)

    output_path = os.path.join(data_dir, "tfrecords/")
    if not tf.gfile.IsDirectory(output_path):
        tf.gfile.MakeDirs(output_path)

    input_size = params["input_size"]
    data_split = params["data_split"]
    num_shards = params["num_shards"]
    data_type = params["data_type"]

    # Data & label directories
    label_dir = os.path.join(data_dir, "labels/")
    data_dir = os.path.join(data_dir, "test-clean/")

    # Split data on files
    train, val, test = split_data(label_dir, data_split, random_seed=0)

    tot_files = len(train) + len(val) + len(test)
    logger.info(f"Total files: {tot_files}")
    logger.info(f"Train/val/test split: {len(train)}/{len(val)}/{len(test)}")
    train_it = slice_iter(data_dir, label_dir, train, input_size)
    val_it = slice_iter(data_dir, label_dir, val, input_size)
    test_it = slice_iter(data_dir, label_dir, test, input_size)

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


def main():
    """Main function to create a TFRecords dataset for VAD."""
    parser = argparse.ArgumentParser(
        description="Create a TFRecords dataset to train VAD TF models."
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        required=True,
        help="Raw dataset directory path.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=1024,
        help="Audio signal input size.",
    )
    parser.add_argument(
        "--data-split",
        type=str,
        default="0.7/0.15",
        help="Train / val dataset split ratios.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=256,
        help="Number of TFRecords files to create.",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default=["train", "val", "test"],
        help="Dataset subsets to create.",
        nargs="+",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Option to debug for a tiny dataset size.",
    )
    args = parser.parse_args()

    params = {
        "input_size": args.input_size,
        "data_split": args.data_split,
        "num_shards": args.num_shards,
        "data_type": args.data_type,
    }

    create_tfrecords(params=params, data_dir=args.data_dir, debug=args.debug)


if __name__ == "__main__":
    main()
