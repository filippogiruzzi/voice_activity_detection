"""Audio feature extraction, raw-data iteration, and the PyTorch VAD dataset.

This module turns raw LibriSpeech ``.flac`` audio plus JSON speech-segment
labels into fixed-length MFCC feature tensors saved as ``.pt`` files, and
exposes a ``Dataset`` / ``DataLoader`` to feed them to the model.
"""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SAMPLE_RATE = 16000
SEQ_LEN = 1024
N_MFCC = 5
FFT_SIZE = 512
HOP_LENGTH = 16
# Feature tensor shape produced by ``extract_features`` for a SEQ_LEN window.
FEAT_SIZE = (16, 65)


def extract_features(
    signal: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    n_fft: int = FFT_SIZE,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Extract MFCC, delta, delta2 and RMS energy features from an audio signal.

    Args:
        signal: Mono audio signal.
        sr: Sampling rate in Hz.
        n_mfcc: Number of MFCC coefficients.
        n_fft: FFT window size.
        hop_length: Hop length between frames.

    Returns:
        Feature array of shape (16, time_steps): 5 MFCC + 5 delta + 5 delta2 + 1 RMS.
    """
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    rmse = librosa.feature.rms(y=signal, frame_length=n_fft, hop_length=hop_length)
    features = np.concatenate([mfcc, mfcc_delta, mfcc_delta2, rmse], axis=0)
    return features.astype(np.float32)


def split_files(
    label_dir: str | Path,
    split: tuple[float, float] = (0.7, 0.15),
    seed: int = 0,
) -> tuple[list[str], list[str], list[str]]:
    """Split the label files into train/val/test stem lists.

    Args:
        label_dir: Directory containing ``<stem>.json`` label files.
        split: (train_fraction, val_fraction); the rest goes to test.
        seed: Random seed for the shuffle.

    Returns:
        Tuple of (train_stems, val_stems, test_stems).
    """
    rng = np.random.default_rng(seed)
    files = [p.stem for p in sorted(Path(label_dir).glob("*.json"))]
    rng.shuffle(files)
    n = len(files)
    n_train, n_val = int(n * split[0]), int(n * split[1])
    return files[:n_train], files[n_train : n_train + n_val], files[n_train + n_val :]


def read_signal(audio_dir: Path, stem: str) -> np.ndarray | None:
    """Read a LibriSpeech ``.flac`` file given its stem.

    The file is resolved as ``audio_dir/<speaker>/<chapter>/<stem>.flac`` where
    the speaker and chapter ids are the first two ``-``-separated fields of the
    stem (``spk-chap-utt``).

    Args:
        audio_dir: Root directory holding the speaker/chapter folder tree.
        stem: File stem in the ``spk-chap-utt`` format.

    Returns:
        The mono audio signal of shape ``(num_samples,)``, or ``None`` if the
        file cannot be read.
    """
    speaker, chapter = stem.split("-")[:2]
    flac_path = audio_dir / speaker / chapter / f"{stem}.flac"
    try:
        signal, _ = sf.read(flac_path)
    except RuntimeError:
        logger.warning(f"Skipped unreadable signal: {flac_path}")
        return None
    return signal


def iter_windows(
    audio_dir: str | Path,
    label_dir: str | Path,
    files: list[str],
    seq_len: int = SEQ_LEN,
) -> Iterator[tuple[np.ndarray, int]]:
    """Yield fixed-length raw audio windows with their binary label.

    Speech/noise regions are derived from the JSON ``speech_segments``, then
    sliced into non-overlapping windows of ``seq_len`` samples.

    Args:
        audio_dir: Directory with ``test-clean`` audio.
        label_dir: Directory with JSON labels.
        files: List of file stems to iterate over.
        seq_len: Window length in samples.

    Yields:
        Tuples of (audio_window, label) where label is 0 (noise) or 1 (speech).
    """
    audio_dir, label_dir = Path(audio_dir), Path(label_dir)
    for stem in files:
        signal = read_signal(audio_dir, stem)
        if signal is None:
            continue
        segments = json.loads((label_dir / f"{stem}.json").read_text())["speech_segments"]

        regions: list[tuple[int, int, int]] = []
        end = 0
        for seg in segments:
            start, seg_end = seg["start_time"], seg["end_time"]
            if end < start:
                regions.append((end, start, 0))
            if start < seg_end:
                regions.append((start, seg_end, 1))
            end = seg_end
        regions.append((end, len(signal) - 1, 0))

        for start, stop, label in regions:
            region = signal[start:stop]
            for i in range(len(region) // seq_len):
                yield region[i * seq_len : (i + 1) * seq_len].astype(np.float32), label


def build_dataset(
    data_dir: str | Path,
    out_dir: str | Path | None = None,
    split: tuple[float, float] = (0.7, 0.15),
    seq_len: int = SEQ_LEN,
    max_files: int | None = None,
    seed: int = 0,
) -> Path:
    """Build train/val/test ``.pt`` datasets from raw LibriSpeech audio and labels.

    Each saved sample is a dict ``{"features": tensor(16, 65), "label": tensor([x])}``.
    To balance the classes, 3 out of 4 speech windows are randomly dropped.

    Args:
        data_dir: Raw dataset root containing ``test-clean/`` and ``labels/``.
        out_dir: Output directory. Defaults to ``<data_dir>/dataset``.
        split: (train_fraction, val_fraction) split ratios.
        seq_len: Audio window length in samples.
        max_files: Optional cap on files per split (useful for quick runs).
        seed: Random seed.

    Returns:
        Path to the output dataset directory.
    """
    rng = np.random.default_rng(seed)
    data_dir = Path(data_dir)
    audio_dir = data_dir / "test-clean"
    label_dir = data_dir / "labels"
    out_dir = Path(out_dir) if out_dir else data_dir / "dataset"

    splits = dict(zip(("train", "val", "test"), split_files(label_dir, split, seed), strict=True))
    if max_files is not None:
        splits = {name: files[:max_files] for name, files in splits.items()}
    logger.info(f"Building dataset from {data_dir} -> {out_dir}")
    logger.info(
        "Files per split - " + ", ".join(f"{name}: {len(files)}" for name, files in splits.items())
    )

    total = 0
    for name, files in splits.items():
        split_dir = out_dir / name
        split_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for window, label in tqdm(iter_windows(audio_dir, label_dir, files, seq_len), desc=name):
            if label == 1 and rng.integers(0, 4) > 0:
                continue
            sample = {
                "features": torch.from_numpy(extract_features(window)),
                "label": torch.tensor([float(label)]),
            }
            torch.save(sample, split_dir / f"sample_{count:08d}.pt")
            count += 1
        total += count
        logger.info(f"{name}: saved {count} samples to {split_dir}")

    logger.info(f"Dataset build complete: {total} samples written to {out_dir}")
    return out_dir


class VADDataset(Dataset):
    """Dataset of ``.pt`` feature/label samples produced by ``build_dataset``."""

    def __init__(self, data_dir: str | Path) -> None:
        """Index all ``.pt`` files inside ``data_dir``.

        Args:
            data_dir: Directory containing ``sample_*.pt`` files.

        Raises:
            FileNotFoundError: If ``data_dir`` contains no ``.pt`` files.
        """
        self.files = sorted(Path(data_dir).glob("*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No .pt samples found in {data_dir}")

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``(features, label)`` pair at ``idx``.

        Args:
            idx: Index of the sample to load.

        Returns:
            Tuple of ``(features, label)`` tensors of shapes ``(16, 65)`` and
            ``(1,)`` respectively.
        """
        sample = torch.load(self.files[idx], weights_only=True)
        return sample["features"], sample["label"]


def create_dataloader(
    data_dir: str | Path,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader over a ``.pt`` dataset split.

    Args:
        data_dir: Directory containing ``sample_*.pt`` files.
        batch_size: Mini-batch size.
        shuffle: Whether to shuffle the samples.
        num_workers: Number of worker processes.

    Returns:
        A configured PyTorch DataLoader.
    """
    dataset = VADDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def main() -> None:
    """CLI entrypoint: build a ``.pt`` dataset from raw LibriSpeech data."""
    parser = argparse.ArgumentParser(description="Build a VAD dataset from raw LibriSpeech audio.")
    parser.add_argument("--data-dir", "-d", required=True, help="Raw dataset directory.")
    parser.add_argument(
        "--out-dir", default=None, help="Output directory (default <data-dir>/dataset)."
    )
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN, help="Audio window length.")
    parser.add_argument(
        "--max-files", type=int, default=None, help="Limit files per split (for quick runs)."
    )
    args = parser.parse_args()
    build_dataset(args.data_dir, args.out_dir, seq_len=args.seq_len, max_files=args.max_files)


if __name__ == "__main__":
    main()
