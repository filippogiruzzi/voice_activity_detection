"""Utilitary functions to visualize audio signals."""
import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf
from loguru import logger

from vad.data_processing.data_iterator import read_json


def plot_signal(sr, signal, labels, signal_id):
    """Utilitary function to plot an audio signal.

    Args:
        sr (int): audio signal sampling rate
        signal (np.ndarray): audio signal
        labels (dict): dictionary of labels (0 or 1)
        signal_id (str): signal id
    """
    logger.info(
        f"Sampling rate = {sr} | Num. points = {len(signal)} | Tot. duration = {len(signal) / sr:.2f} s"
    )
    plt.figure(figsize=(15, 10))
    sns.set()
    sns.lineplot(x=[i / sr for i in range(len(signal))], y=signal)

    start, end = 0, 0
    for seg in labels["speech_segments"]:
        plt.axvspan(end, seg["start_time"] / sr, alpha=0.5, color="r")
        start, end = seg["start_time"] / sr, seg["end_time"] / sr
        plt.axvspan(start, end, alpha=0.5, color="g")
    plt.axvspan(end, (len(signal) - 1) / sr, alpha=0.5, color="r")

    plt.title(f"Sample number {signal_id} with speech in green", size=20)
    plt.xlabel("Time (s)", size=20)
    plt.ylabel("Amplitude", size=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.show()


def visualize_audio_signals(data_dir):
    """Utilitary function to visualize audio signals with labels.

    Args:
        data_dir (str): path to raw dataset directory
    """
    test_data_dir = os.path.join(data_dir, "test-clean/")
    label_dir = os.path.join(data_dir, "labels/")

    files = [x.split(".")[0] for x in os.listdir(label_dir) if "json" in x]

    for fn in files:
        # Read .flac
        logger.info("Reading .flac file ...")
        fn_ids = fn.split("-")
        flac_fp = os.path.join(test_data_dir, fn_ids[0], fn_ids[1], f"{fn}.flac")
        signal, sr = sf.read(flac_fp)

        # Read .json
        logger.info("Reading .json file ...")
        labels = read_json(label_dir, fn)

        # Plot
        logger.info("Plotting signal ...")
        plot_signal(sr, signal, labels, fn)


def main():
    """Main function to visualize audio signals."""
    parser = argparse.ArgumentParser(
        description="Visualize raw audio signals and labels."
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        required=True,
        help="Raw dataset directory path.",
    )
    args = parser.parse_args()

    visualize_audio_signals(data_dir=args.data_dir)


if __name__ == "__main__":
    main()
