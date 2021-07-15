import json
import os
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import soundfile as sf
import tensorflow as tf
from absl import app, flags
from loguru import logger

from vad.data_processing.feature_extraction import extract_features

flags.DEFINE_string(
    "data_dir",
    "/home/filippo/datasets/LibriSpeech/test-clean/",
    "path to data directory",
)
flags.DEFINE_string(
    "out_dir",
    "/home/filippo/datasets/LibriSpeech/labels/",
    "output directory to record labels",
)
flags.DEFINE_string(
    "exported_model",
    "/home/filippo/datasets/vad_data/tfrecords/models/resnet1d/inference/exported/",
    "path to pretrained TensorFlow exported model",
)
flags.DEFINE_boolean("viz", False, "visualize prediction")
FLAGS = flags.FLAGS


def visualize_predictions(signal, fn, preds):
    fig = plt.figure(figsize=(15, 10))
    sns.set()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([i / 16000 for i in range(len(signal))], signal)
    for predictions in preds:
        color = "r" if predictions[2] == 0 else "g"
        ax.axvspan(
            (predictions[0]) / 16000, predictions[1] / 16000, alpha=0.5, color=color
        )
    plt.title(f"Prediction on signal {fn}, speech in green", size=20)
    plt.xlabel("Time (s)", size=20)
    plt.ylabel("Amplitude", size=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.show()


def file_iter(data_dir):
    for base_dir_id in os.listdir(data_dir):
        base_dir = os.path.join(data_dir, base_dir_id)
        for sub_dir_id in os.listdir(base_dir):
            sub_dir = os.path.join(base_dir, sub_dir_id)
            flac_files = [x for x in os.listdir(sub_dir) if "flac" in x]
            for fn in flac_files:
                fp = os.path.join(data_dir, base_dir, sub_dir, fn)
                try:
                    signal, sr = sf.read(fp)
                except RuntimeError:
                    logger.warning("!!! Skipped signal !!!")
                    continue
                yield signal, fn


def main(_):
    np.random.seed(0)
    file_it = file_iter(FLAGS.data_dir)
    if not tf.gfile.IsDirectory(FLAGS.out_dir):
        tf.gfile.MakeDirs(FLAGS.out_dir)

    # TensorFlow inputs
    features_input_ph = tf.placeholder(shape=(16, 65), dtype=tf.float32)
    features_input_op = tf.transpose(features_input_ph, perm=[1, 0])
    features_input_op = tf.expand_dims(features_input_op, axis=0)

    # TensorFlow exported model
    speech_predictor = tf.contrib.predictor.from_saved_model(
        export_dir=FLAGS.exported_model
    )
    init = tf.initializers.global_variables()
    classes = ["Noise", "Speech"]

    # Iterate though test data
    with tf.Session() as sess:
        for signal, fn in file_it:
            sess.run(init)
            logger.info(f"Prediction on file {fn} ...")
            signal_input = deque(signal[:1024].tolist(), maxlen=1024)

            labels = {"speech_segments": []}
            preds, pred_time = [], []
            pointer = 1024
            while pointer < len(signal):
                start = time.time()
                # Preprocess signal & extract features
                signal_to_process = np.copy(signal_input)
                signal_to_process = np.float32(signal_to_process)
                signal_to_process = np.add(signal_to_process, 1.0)
                signal_to_process = np.divide(signal_to_process, 2.0)
                features = extract_features(
                    signal_to_process, freq=16000, n_mfcc=5, size=512, step=16
                )

                # Prediction
                features_input = sess.run(
                    features_input_op, feed_dict={features_input_ph: features}
                )
                speech_prob = speech_predictor({"features_input": features_input})[
                    "speech"
                ][0]
                speech_pred = classes[int(np.round(speech_prob))]

                # Time prediction & processing
                end = time.time()
                dt = end - start
                pred_time.append(dt)
                if FLAGS.viz:
                    logger.info(
                        f"Prediction = {speech_pred} | proba = {speech_prob[0]:.2f} | time = {dt:.2f} s"
                    )

                # For visualization
                preds.append([pointer - 1024, pointer, np.round(speech_prob)])

                # For label recording
                if np.round(speech_prob) > 0:
                    labels["speech_segments"].append(
                        {"start_time": pointer - 1024, "end_time": pointer}
                    )

                # Update signal segment
                signal_input.extend(signal[pointer + 1 : pointer + 1 + 1024])
                pointer += 1024 + 1

            logger.info(f"Average prediction time = {np.mean(pred_time) * 1e3:.2f} ms")

            # Visualization
            if FLAGS.viz:
                visualize_predictions(signal, fn, preds)

            # Record labels to .json
            if not FLAGS.viz:
                base_name = fn.split(".")[0]
                out_fn = f"{base_name}.json"
                out_fp = os.path.join(FLAGS.out_dir, out_fn)
                with open(out_fp, "w") as f:
                    json.dump(labels, f)

                nb_preds = len(labels["speech_segments"])
                output_dir = FLAGS.out_dir
                logger.info(f"{nb_preds} predictions recorded to {output_dir}")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)
