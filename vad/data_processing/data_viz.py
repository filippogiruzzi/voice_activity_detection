import argparse
import os

import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf

from vad.data_processing.data_iterator import read_json


def plot_signal(sr, signal, labels, signal_id):
    print('Sampling rate = {} | Num. points = {} | Tot. duration = {:.2f} s'.format(sr, len(signal), len(signal) / sr))
    plt.figure(figsize=(15, 10))
    sns.set()
    sns.lineplot(x=[i / sr for i in range(len(signal))], y=signal)

    start, end = 0, 0
    for seg in labels['speech_segments']:
        plt.axvspan(end, seg['start_time'] / sr, alpha=0.5, color='r')
        start, end = seg['start_time'] / sr, seg['end_time'] / sr
        plt.axvspan(start, end, alpha=0.5, color='g')
    plt.axvspan(end, (len(signal) - 1) / sr, alpha=0.5, color='r')

    plt.title('Sample number {} with speech in green'.format(signal_id), size=20)
    plt.xlabel('Time (s)', size=20)
    plt.ylabel('Amplitude', size=20)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='visualize raw data')
    parser.add_argument('--data-dir', type=str, default='/home/filippo/datasets/LibriSpeech/')
    args = parser.parse_args()

    data_dir = os.path.join(args.data_dir, 'test-clean/')
    label_dir = os.path.join(args.data_dir, 'labels/')

    files = [x.split('.')[0] for x in os.listdir(label_dir) if 'json' in x]

    for fn in files:
        # Read .flac
        print('\nReading .flac file ...')
        fn_ids = fn.split('-')
        flac_fp = os.path.join(data_dir, fn_ids[0], fn_ids[1], '{}.flac'.format(fn))
        signal, sr = sf.read(flac_fp)

        # Read .json
        print('Reading .json file ...')
        labels = read_json(label_dir, fn)

        # Plot
        print('Plotting signal ...')
        plot_signal(sr, signal, labels, fn)


if __name__ == '__main__':
    main()
