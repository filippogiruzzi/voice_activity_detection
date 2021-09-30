"""Training TF input pipeline."""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from loguru import logger

FEAT_SIZE = (16, 65)


def get_dataset(
    tfrecords,
    batch_size,
    epochs,
    input_size=1024,
    n_classes=1,
    shuffle=False,
    fake_input=False,
):
    """Reads a TFRecord dataset and builds a TF dataset to feed to a TF model.

    Args:
        tfrecords (list): TFRecord files paths
        batch_size (int): batch size
        epochs (int): number of epochs
        input_size (int, optional): sub audio segment window size. Defaults to 1024.
        n_classes (int, optional): number of classes. Defaults to 1.
        shuffle (bool, optional): shuffle dataset or not. Defaults to False.
        fake_input (bool, optional): build a fake dataset for debugging. Defaults to False.
    """

    def parse_func(example_proto):
        """Parse a TF example to build TF inputs & labels.

        Args:
            example_proto (tf.core.example.example_pb2.Example): TF training example

        Returns:
            features, labels (dict, dict): dictionaries of input features & associated labels
        """
        feature_dict = {
            "signal/id": tf.FixedLenFeature([], tf.string),
            "segment/start": tf.FixedLenFeature([], tf.int64),
            "segment/end": tf.FixedLenFeature([], tf.int64),
            "subsegment/id": tf.FixedLenFeature([], tf.int64),
            "subsegment/length": tf.FixedLenFeature([], tf.int64),
            "subsegment/signal": tf.FixedLenFeature([input_size], tf.float32),
            "subsegment/features": tf.FixedLenFeature(
                [FEAT_SIZE[0] * FEAT_SIZE[1]], tf.float32
            ),
            "subsegment/label": tf.FixedLenFeature([], tf.int64),
        }
        parsed_feature = tf.parse_single_example(example_proto, feature_dict)

        features, labels = {}, {}
        for key, val in parsed_feature.items():
            if key == "subsegment/signal":
                val = tf.cast(val, dtype=tf.float32)
                features[key] = val
            elif key == "subsegment/features":
                val = tf.cast(val, dtype=tf.float32)
                # @ToDo: generate metadata files with tfrecords
                val = tf.reshape(val, [FEAT_SIZE[0], FEAT_SIZE[1]])
                features[key] = val
            elif key == "subsegment/label":
                if n_classes > 1:
                    val = tf.one_hot(val, depth=n_classes)
                else:
                    val = tf.expand_dims(val, axis=-1)
                val = tf.cast(val, dtype=tf.float32)
                labels[key] = val
            else:
                features[key] = val
        return features, labels

    files = tf.data.Dataset.list_files(tfrecords)
    dataset = files.apply(
        tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=10)
    )
    if shuffle:
        dataset = dataset.apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size=256, count=epochs)
        )
    else:
        dataset = dataset.repeat(epochs)
    dataset = dataset.map(parse_func, num_parallel_calls=8)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=256)
    if fake_input:
        dataset = dataset.take(1).cache().repeat()
    return dataset


def data_input_fn(
    tfrecords,
    batch_size,
    epochs,
    input_size=1024,
    n_classes=1,
    subsample=False,
    shuffle=False,
    fake_input=False,
):
    """Input pipeline function to feed to a TF estimator for training / evaluation.

    Args:
        tfrecords (list): TFRecord files paths
        batch_size (int): batch size
        epochs (int): number of epochs
        input_size (int, optional): sub audio segment window size. Defaults to 1024.
        n_classes (int, optional): number of classes. Defaults to 1.
        subsample (bool, optional): subsample the signal or not. Defaults to False.
        shuffle (bool, optional): shuffle dataset or not. Defaults to False.
        fake_input (bool, optional): build a fake dataset for debugging. Defaults to False.
    """

    def _input_fn():
        dataset = get_dataset(
            tfrecords,
            batch_size,
            epochs,
            input_size,
            n_classes,
            shuffle,
            fake_input,
        )

        it = dataset.make_one_shot_iterator()
        next_batch = it.get_next()

        signal_input = next_batch[0]["subsegment/signal"]
        features_input = next_batch[0]["subsegment/features"]
        label = next_batch[1]["subsegment/label"]

        # Subsample signal
        if subsample:
            signal_input = tf.strided_slice(
                signal_input,
                begin=[0, 0, 0],
                end=[batch_size, input_size, 1],
                strides=[1, 8, 1],
            )

        features, labels = {}, {}
        features["signal_input"] = signal_input
        features["features_input"] = tf.transpose(features_input, perm=[0, 2, 1])
        labels["label"] = label
        return features, labels

    return _input_fn


def visualize_input_pipeline(data_dir, data_type="train", n_classes=1):
    """Utilitary function to visualize the data through the TF input pipeline.

    Args:
        data_dir (str): path to TFRecords dataset directory
        data_type (str, optional): dataset subset to visualize. Defaults to "train"
        n_classes (int, optional): number of output classes. Defaults to 1
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    classes = ["Noise", "Speech"]
    tfrecords = glob.glob(f"{data_dir}{data_type}/*.tfrecord")
    dataset = get_dataset(
        tfrecords,
        batch_size=32,
        epochs=1,
        input_size=1024,
        n_classes=n_classes,
        shuffle=False,
        fake_input=False,
    )
    logger.info(f"Dataset out types {dataset.output_types}")

    batch = dataset.make_one_shot_iterator().get_next()
    sess = tf.Session()
    try:
        batch_nb = 0
        while True:
            data = sess.run(batch)
            batch_nb += 1

            signal_id = data[0]["signal/id"]
            start = data[0]["segment/start"]
            end = data[0]["segment/end"]
            sub_id = data[0]["subsegment/id"]
            length = data[0]["subsegment/length"]
            signal_input = data[0]["subsegment/signal"]
            label = data[1]["subsegment/label"]

            logger.info(f"Batch nb {batch_nb}")
            for i in range(len(signal_input)):
                class_name = classes[int(np.round(label[i]))]
                logger.info(f"Class: {class_name}")
                logger.info(
                    signal_id[i].decode(), start[i], end[i], sub_id[i], length[i]
                )

                # Plot signal
                plt.figure(figsize=(15, 10))
                sns.set()
                sns.lineplot(
                    x=[i for i in range(len(signal_input[i]))], y=signal_input[i]
                )
                plt.title(f"Signal = {class_name}", size=20)
                plt.xlabel("Time (num. points)", size=20)
                plt.ylabel("Amplitude", size=20)
                plt.xticks(size=15)
                plt.yticks(size=15)
                plt.show()

    except tf.errors.OutOfRangeError:
        pass


def main():
    """Main function to visualize data through the TF data pipeline."""
    parser = argparse.ArgumentParser(
        description="Visualize data through the TF input pipeline."
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        required=True,
        help="TFRecords dataset directory path.",
    )
    parser.add_argument(
        "--n-classes",
        "-n",
        type=int,
        default=1,
        choices=[1],
        help="Number of output classes.",
    )
    parser.add_argument(
        "--data-type",
        "-t",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset subset to visualize.",
    )
    args = parser.parse_args()

    visualize_input_pipeline(
        data_dir=args.data_dir,
        data_type=args.data_type,
        n_classes=args.n_classes,
    )


if __name__ == "__main__":
    main()
