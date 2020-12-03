import argparse
import os

import numpy as np
import json
import soundfile as sf
from tabulate import tabulate


def split_data(label_dir, split="0.7/0.15", random_seed=0):
    np.random.seed(random_seed)
    splits = [float(x) for x in split.split("/")]
    assert sum(splits) < 1.0, "Wrong split values"

    files = [fn.split(".")[0] for fn in os.listdir(label_dir) if "json" in fn]
    np.random.shuffle(files)

    n_train, n_val = int(len(files) * splits[0]), int(len(files) * splits[1])
    train_fns, val_fns, test_fns = (
        files[:n_train],
        files[n_train : n_train + n_val],
        files[n_train + n_val :],
    )
    return train_fns, val_fns, test_fns


def read_json(data_dir, fn):
    fp = os.path.join(data_dir, fn + ".json")
    with open(fp, "r") as f:
        labels = json.load(f)
    return labels


def file_iter(data_dir, label_dir, files):
    for fn in files:
        fn_ids = fn.split("-")
        flac_fp = os.path.join(data_dir, fn_ids[0], fn_ids[1], "{}.flac".format(fn))
        try:
            signal, _ = sf.read(flac_fp)
        except RuntimeError:
            print("!!! Skipped signal !!!")
            continue
        labels = read_json(label_dir, fn)
        yield signal, labels["speech_segments"], fn


def data_iter(data_dir, label_dir, files):
    file_it = file_iter(data_dir, label_dir, files)
    for signal, labels, file_id in file_it:
        start, end = 0, 0
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
            start, end = next_start, next_end
        # yield end of signal as noise
        yield {
            "id": file_id,
            "start": end,
            "end": len(signal) - 1,
            "segment": signal[end:],
            "label": 0,
        }


def slice_iter(data_dir, label_dir, files, seq_len=1024):
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


def main():
    parser = argparse.ArgumentParser(description="data iterator to loop through data")
    parser.add_argument(
        "--data-dir", type=str, default="/home/filippo/datasets/LibriSpeech/"
    )
    args = parser.parse_args()

    data_dir = os.path.join(args.data_dir, "test-clean/")
    label_dir = os.path.join(args.data_dir, "labels/")

    train_fns, val_fns, test_fns = split_data(label_dir, split="0.7/0.15")
    print(
        "\nTrain: {} | Val: {} | Test: {}".format(
            len(train_fns), len(val_fns), len(test_fns)
        )
    )

    # Some stats to get the number of noise & speech segments, min & max values
    data_dict = {"train": train_fns, "val": val_fns, "test": test_fns}
    stats_dict = {"train": [0, 0, 0, 0], "val": [0, 0, 0, 0], "test": [0, 0, 0, 0]}
    to_print = []
    seq_len = 1024
    for data_type, data_files in data_dict.items():
        data_it = slice_iter(data_dir, label_dir, data_files, seq_len)
        for data in data_it:
            # print(data['file_id'],
            #       data['label'],
            #       data['start'],
            #       data['end'],
            #       data['sub_segment_id'],
            #       data['sub_segment_len'])

            # Drop 1/4th of speech signals to balance data
            # if data['label'] == 1:
            #     drop = np.random.randint(0, 4)
            #     if drop > 0:
            #         continue

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

    print(
        tabulate(
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
    )


if __name__ == "__main__":
    main()
