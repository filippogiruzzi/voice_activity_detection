"""Automatic data annotations for LibriSpeech dataset.

Uses a pre-trained VAD model to annotate audio signal automatically.
"""
import argparse
import json
import os
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import soundfile as sf
import tensorflow as tf
from loguru import logger

from vad.data_processing.feature_extraction import extract_features


def visualize_predictions(signal, fn, preds):
    """Utilitary function to visualize predictions over audio signals.

    Args:
        signal (np.ndarray): audio signal
        fn (str): file name
        preds (np.ndarray): VAD predictions
    """
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
    """File iterator to loop over audio signal files.

    Args:
        data_dir (str): path to directory containing audio signal files

    Yields:
        signal, fn (np.ndarray, str): audio signal and file name
    """
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


def automatic_labeling(data_dir, exported_model, visualize=False):
    """Run automatic labeling over a given dataset of raw audio signals, given a pre-trained VAD model.

    Args:
        data_dir (str): path to raw dataset directory
        exported_model (str): path to exported pre-trained TF model directory
        visualize (bool, optional): option to visualize automatic labeling. Defaults to False.
    """
    np.random.seed(0)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    tf.logging.set_verbosity(tf.logging.INFO)

    test_data_dir = os.path.join(data_dir, "test-clean/")
    labels_dir = os.path.join(data_dir, "labels/")

    file_it = file_iter(test_data_dir)
    if not tf.gfile.IsDirectory(labels_dir):
        tf.gfile.MakeDirs(labels_dir)

    # TensorFlow inputs
    features_input_ph = tf.placeholder(shape=(16, 65), dtype=tf.float32)
    features_input_op = tf.transpose(features_input_ph, perm=[1, 0])
    features_input_op = tf.expand_dims(features_input_op, axis=0)

    # TensorFlow exported model
    speech_predictor = tf.contrib.predictor.from_saved_model(export_dir=exported_model)
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
                if visualize:
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
            if visualize:
                visualize_predictions(signal, fn, preds)

            # Record labels to .json
            if not visualize:
                base_name = fn.split(".")[0]
                out_fn = f"{base_name}.json"
                out_fp = os.path.join(labels_dir, out_fn)
                with open(out_fp, "w") as f:
                    json.dump(labels, f)

                nb_preds = len(labels["speech_segments"])
                logger.info(f"{nb_preds} predictions recorded to {labels_dir}")


def main(_):
    """Main function to run automatic data annotation."""
    parser = argparse.ArgumentParser(
        description="Run Voice Activity Detection CNN inference over audio signals."
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        required=True,
        help="Raw dataset directory path.",
    )
    parser.add_argument(
        "--exported-model",
        type=str,
        required=True,
        help="Path to pre-trained exported TF model.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Visualize automatic labeling.",
    )
    args = parser.parse_args()

    automatic_labeling(
        data_dir=args.data_dir,
        exported_model=args.exported_model,
        visualize=args.visualize,
    )


if __name__ == "__main__":
    main()
