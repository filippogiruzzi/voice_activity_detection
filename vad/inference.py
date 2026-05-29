"""Run VAD inference on raw audio with a sliding window, plus visualization."""

import argparse
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from loguru import logger

from vad.data import SAMPLE_RATE, SEQ_LEN, extract_features, read_signal, split_files
from vad.model import ModelConfig, Resnet1D

CLASSES = ("Noise", "Speech")


def load_model(
    checkpoint: str | Path,
    config: ModelConfig | None = None,
    device: torch.device | None = None,
) -> Resnet1D:
    """Load a trained ``Resnet1D`` from a state-dict checkpoint in eval mode.

    Args:
        checkpoint: Path to a saved ``state_dict``.
        config: Model architecture used to rebuild the network. Must match the
            architecture used to train ``checkpoint``. Defaults to
            ``ModelConfig()`` when omitted.
        device: Target device. Defaults to CUDA when available, else CPU.

    Returns:
        The reconstructed model on ``device`` in evaluation mode.
    """
    config = config or ModelConfig()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Resnet1D(config)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_signal(
    model: Resnet1D,
    signal: np.ndarray,
    seq_len: int = SEQ_LEN,
    stride: int = 1,
    device: torch.device | None = None,
) -> list[list[float]]:
    """Run sliding-window VAD over a single signal.

    Args:
        model: Trained VAD model in eval mode.
        signal: Mono audio signal.
        seq_len: Window length in samples.
        stride: Gap between consecutive windows.
        device: Compute device. Inferred from the model when None.

    Returns:
        List of ``[start_sample, end_sample, prediction]`` triplets, where
        prediction is the rounded speech probability (0.0 or 1.0).
    """
    device = device or next(model.parameters()).device
    window = deque(signal[:seq_len].tolist(), maxlen=seq_len)
    preds: list[list[float]] = []
    pointer = seq_len

    with torch.no_grad():
        while pointer < len(signal):
            # Copy the rolling window into a contiguous float32 array for librosa.
            features = extract_features(np.asarray(window, dtype=np.float32))
            features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(device)
            prob = torch.sigmoid(model(features_tensor)).cpu().item()
            preds.append([pointer - seq_len, pointer, float(np.round(prob))])
            window.extend(signal[pointer + stride : pointer + stride + seq_len])
            pointer += seq_len + stride

    return preds


def smooth_predictions(preds: list[list[float]]) -> list[list[float]]:
    """Smooth predictions with 3-window majority voting and a hangover scheme.

    Args:
        preds: List of ``[start, end, prediction]`` triplets.

    Returns:
        Smoothed list of ``[start, end, prediction]`` triplets.
    """
    smoothed: list[list[float]] = []
    for i in range(2, len(preds), 3):
        cur = preds[i]
        if cur[2] == preds[i - 1][2] == preds[i - 2][2]:
            smoothed.append([preds[i - 2][0], cur[1], cur[2]])
        elif smoothed:
            smoothed.append([preds[i - 2][0], cur[1], smoothed[-1][2]])
        else:
            smoothed.append([preds[i - 2][0], cur[1], 0.0])

    n = 0
    while n < len(smoothed):
        if smoothed[n][2] == 1:
            if n > 0:
                smoothed[n - 1][2] = 1
            if n < len(smoothed) - 1:
                smoothed[n + 1][2] = 1
            n += 2
        else:
            n += 1
    return smoothed


def visualize_predictions(
    signal: np.ndarray,
    fn: str,
    preds: list[list[float]],
    sr: int = SAMPLE_RATE,
) -> None:
    """Plot the waveform with predicted speech/noise regions overlaid.

    Predicted speech spans are shaded green and noise spans red.

    Args:
        signal: Mono audio signal of shape ``(num_samples,)``.
        fn: File name (or stem) used in the plot title.
        preds: List of ``[start_sample, end_sample, prediction]`` triplets.
        sr: Sampling rate in Hz, used to convert sample indices to seconds.
    """
    fig = plt.figure(figsize=(15, 10))
    sns.set()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot([i / sr for i in range(len(signal))], signal)
    for start, end, pred in preds:
        ax.axvspan(start / sr, end / sr, alpha=0.5, color="g" if pred == 1 else "r")
    plt.title(f"Prediction on signal {fn}, speech in green", size=20)
    plt.xlabel("Time (s)", size=20)
    plt.ylabel("Amplitude", size=20)
    plt.show()


def run_inference(
    data_dir: str | Path,
    checkpoint: str | Path,
    config: ModelConfig | None = None,
    seq_len: int = SEQ_LEN,
    stride: int = 1,
    smoothing: bool = False,
    max_files: int | None = None,
) -> None:
    """Run VAD inference over the test split and visualize the predictions.

    Args:
        data_dir: Raw dataset root with ``test-clean/`` and ``labels/``.
        checkpoint: Path to a trained state-dict checkpoint.
        config: Model architecture, which must match the one used to train
            ``checkpoint``. Defaults to ``ModelConfig()`` when omitted.
        seq_len: Sliding-window length in samples.
        stride: Gap between consecutive windows.
        smoothing: Whether to smooth the predictions before plotting.
        max_files: Optional cap on the number of test files to process.
    """
    data_dir = Path(data_dir)
    audio_dir = data_dir / "test-clean"
    label_dir = data_dir / "labels"
    model = load_model(checkpoint, config=config)
    logger.info(f"Loaded model from {checkpoint}")

    _, _, test_files = split_files(label_dir, split=(0.7, 0.15), seed=0)
    if max_files is not None:
        test_files = test_files[:max_files]
    logger.info(f"Running inference on {len(test_files)} file(s) from {audio_dir}")

    processed = 0
    for stem in test_files:
        signal = read_signal(audio_dir, stem)
        if signal is None:
            continue
        logger.info(f"Prediction on file {stem} ...")
        start = time.time()
        preds = predict_signal(model, signal, seq_len=seq_len, stride=stride)
        logger.info(f"Predicted {len(preds)} windows in {time.time() - start:.2f}s")
        if smoothing:
            preds = smooth_predictions(preds)
        visualize_predictions(signal, stem, preds)
        processed += 1

    logger.info(f"Inference complete: processed {processed}/{len(test_files)} file(s)")


def main() -> None:
    """CLI entrypoint for VAD inference."""
    parser = argparse.ArgumentParser(description="Run VAD inference over raw audio signals.")
    parser.add_argument("--data-dir", "-d", required=True, help="Raw dataset directory.")
    parser.add_argument("--checkpoint", "-c", required=True, help="Trained state-dict checkpoint.")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN, help="Sliding-window length.")
    parser.add_argument("--stride", type=int, default=1, help="Window stride.")
    parser.add_argument("--smoothing", action="store_true", help="Smooth predictions.")
    parser.add_argument("--max-files", type=int, default=None, help="Cap on test files.")
    parser.add_argument(
        "--n-filters", type=int, nargs="+", default=[32, 64, 128, 128], help="Conv filters."
    )
    parser.add_argument(
        "--fc-units", type=int, nargs="+", default=[2048, 2048], help="FC hidden units."
    )
    args = parser.parse_args()

    config = ModelConfig(n_filters=tuple(args.n_filters), fc_units=tuple(args.fc_units))

    run_inference(
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        config=config,
        seq_len=args.seq_len,
        stride=args.stride,
        smoothing=args.smoothing,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
