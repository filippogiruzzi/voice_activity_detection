import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

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
    def parse_func(example_proto):
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
    def _input_fn():
        dataset = get_dataset(
            tfrecords, batch_size, epochs, input_size, n_classes, shuffle, fake_input
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


def main():
    parser = argparse.ArgumentParser(description="visualize input pipeline")
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default="/home/filippo/datasets/LibriSpeech/tfrecords/",
    )
    parser.add_argument("--n-classes", "-n", type=int, default=1)
    parser.add_argument("--data-type", "-t", type=str, default="train")
    args = parser.parse_args()

    classes = ["Noise", "Speech"]
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tfrecords = glob.glob("{}{}/*.tfrecord".format(args.data_dir, args.data_type))
    dataset = get_dataset(
        tfrecords,
        batch_size=32,
        epochs=1,
        input_size=1024,
        n_classes=args.n_classes,
        shuffle=False,
        fake_input=False,
    )
    print("\nDataset out types {}".format(dataset.output_types))

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

            print("\nBatch nb {}".format(batch_nb))
            for i in range(len(signal_input)):
                class_name = classes[int(np.round(label[i]))]
                print("\nClass: {}".format(class_name))
                print(signal_id[i].decode(), start[i], end[i], sub_id[i], length[i])

                # Plot signal
                plt.figure(figsize=(15, 10))
                sns.set()
                sns.lineplot(
                    x=[i for i in range(len(signal_input[i]))], y=signal_input[i]
                )
                plt.title("Signal = {}".format(class_name), size=20)
                plt.xlabel("Time (num. points)", size=20)
                plt.ylabel("Amplitude", size=20)
                plt.xticks(size=15)
                plt.yticks(size=15)
                plt.show()

    except tf.errors.OutOfRangeError:
        pass


if __name__ == "__main__":
    main()
