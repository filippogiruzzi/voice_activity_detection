"""
Global data iterator that reads & pre-processed input data to be recorded as a TFRecord dataset.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger
from tabulate import tabulate


def split_data(label_dir, split="0.7/0.15", random_seed=0):
    """Split a directory of files into train / val / test files.

    Args:
        label_dir (str): path to directory containing label files
        split (str, optional): train / val split. Defaults to "0.7/0.15".
        random_seed (int, optional): random see value. Defaults to 0.

    Returns:
        train_fns, val_fns, test_fns (list, list, list): train / val / test files lists
    """
    np.random.seed(random_seed)
    splits = [float(x) for x in split.split("/")]
    assert sum(splits) < 1.0, "Wrong split values"

    files = [fn.stem for fn in Path(label_dir).glob("*.json")]
    np.random.shuffle(files)

    n_train, n_val = int(len(files) * splits[0]), int(len(files) * splits[1])
    train_fns, val_fns, test_fns = (
        files[:n_train],
        files[n_train : n_train + n_val],
        files[n_train + n_val :],
    )
    return train_fns, val_fns, test_fns


def read_json(data_dir, fn):
    """Utilitary function to read a .json file.

    Args:
        data_dir (str): path to parent directory containing the file to read
        fn (str): name of the file to read, without the extension

    Returns:
        labels (dict): .json file content
    """
    fp = os.path.join(data_dir, fn + ".json")
    with open(fp, "r") as f:
        labels = json.load(f)
    return labels


def file_iter(data_dir, label_dir, files):
    """File iterator to yield signal+label file.

    Args:
        data_dir (str): path to main data directory containing audio files
        label_dir (str): path to directory containing label files
        files (list): list of files to read

    Yields:
        signal, label, fn (np.ndarray, list, str): audio signal, label list and file name without extension
    """
    for fn in files:
        fn_ids = fn.split("-")
        flac_fp = os.path.join(data_dir, fn_ids[0], fn_ids[1], f"{fn}.flac")
        try:
            signal, _ = sf.read(flac_fp)
        except RuntimeError:
            logger.warning("!!! Skipped signal !!!")
            continue

        labels = read_json(label_dir, fn)
        label = labels["speech_segments"]

        yield signal, label, fn


def data_iter(data_dir, label_dir, files):
    """Data iterator to yield an audio segment with a constant label (noise or audio).

    Args:
        data_dir (str): path to main data directory containing audio files
        label_dir (str): path to directory containing label files
        files (list): list of files to read

    Yields:
        (dict): dict containing info about each audio segment
    """
    file_it = file_iter(data_dir, label_dir, files)
    for signal, labels, file_id in file_it:
        # start, end = 0, 0
        end = 0
        for seg in labels:
            next_start, next_end = seg["start_time"], seg["end_time"]
            # yield noise segment
            if end < next_start:
                yield {
                    "id": file_id,
                    "start": end,
                    "end": next_start,
                    "segment": signal[end:next_start],
                    "label": 0,
                }
            # yield speech segment
            if next_start < next_end:
                yield {
                    "id": file_id,
                    "start": next_start,
                    "end": next_end,
                    "segment": signal[next_start:next_end],
                    "label": 1,
                }
            # start, end = next_start, next_end
            end = next_end
        # yield end of signal as noise
        yield {
            "id": file_id,
            "start": end,
            "end": len(signal) - 1,
            "segment": signal[end:],
            "label": 0,
        }


def slice_iter(data_dir, label_dir, files, seq_len=1024):
    """Data iterator to yield and fixed length audio segment, given a sliding window size.

    Args:
        data_dir (str): path to main data directory containing audio files
        label_dir (str): path to directory containing label files
        files (list): list of files to read
        seq_len (int, optional): size of the sliding window. Defaults to 1024.

    Yields:
        (dict): dict containing info about each audio segment
    """
    data_it = data_iter(data_dir, label_dir, files)
    # Slice segment with length seq_len & no overlap @Todo: add overlap with stride s < seq_len
    for data in data_it:
        if len(data["segment"]) < seq_len:
            continue
        counter = len(data["segment"]) // seq_len
        for n in range(counter):
            sub_segment = data["segment"][n * seq_len : (n + 1) * seq_len]
            yield {
                "file_id": data["id"],
                "start": data["start"],
                "end": data["end"],
                "sub_segment": sub_segment,
                "sub_segment_id": n,
                "sub_segment_len": seq_len,
                "label": data["label"],
            }


def run_data_iterator(data_dir):
    """Utilitary function to use the data iterator to loop through the raw data and compute some statistics.

    Args:
        data_dir (str): path to raw dataset directory
    """
    test_data_dir = os.path.join(data_dir, "test-clean/")
    label_dir = os.path.join(data_dir, "labels/")

    train_fns, val_fns, test_fns = split_data(label_dir, split="0.7/0.15")
    logger.info(
        f"Train: {len(train_fns)} | Val: {len(val_fns)} | Test: {len(test_fns)}"
    )

    # Some stats to get the number of noise & speech segments, min & max values
    data_dict = {"train": train_fns, "val": val_fns, "test": test_fns}
    stats_dict = {"train": [0, 0, 0, 0], "val": [0, 0, 0, 0], "test": [0, 0, 0, 0]}
    to_print = []
    seq_len = 1024
    for data_type, data_files in data_dict.items():
        data_it = slice_iter(test_data_dir, label_dir, data_files, seq_len)
        for data in data_it:
            stats_dict[data_type][data["label"]] += 1
            mini, maxi = stats_dict[data_type][2:]
            if np.min(data["sub_segment"]) < mini:
                stats_dict[data_type][2] = np.min(data["sub_segment"])
            if np.max(data["sub_segment"]) > maxi:
                stats_dict[data_type][3] = np.max(data["sub_segment"])

        to_print.append(
            [
                data_type,
                seq_len,
                stats_dict[data_type][0],
                stats_dict[data_type][1],
                stats_dict[data_type][2],
                stats_dict[data_type][3],
            ]
        )

    tabulate_val = tabulate(
        headers=[
            "Data type",
            "Seg. length",
            "Num. noise",
            "Num. speech",
            "Min",
            "Max",
        ],
        tabular_data=to_print,
        tablefmt="simple",
    )
    logger.info(f"\n{tabulate_val}")


def main():
    """Temporary main function. Should be move to tests."""
    parser = argparse.ArgumentParser(
        description="Loop through the raw data and compute some statistics."
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        required=True,
        help="Raw dataset directory path.",
    )
    args = parser.parse_args()

    run_data_iterator(data_dir=args.data_dir)


if __name__ == "__main__":
    main()
