"""Train, evaluate, and export a 1D ResNet VAD model."""

import argparse
from pathlib import Path

import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vad.data import FEAT_SIZE, create_dataloader
from vad.model import ModelConfig, Resnet1D


def get_device() -> torch.device:
    """Return the best available compute device.

    Returns:
        The first available device among CUDA, Apple MPS, and CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate a model over a data loader.

    Args:
        model: The model to evaluate.
        loader: DataLoader yielding ``(features, labels)`` batches.
        criterion: Loss function applied to the raw logits.
        device: Device on which to run the forward pass.

    Returns:
        Tuple of ``(average_loss, accuracy)``. Both are ``0.0`` when the loader
        is empty.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            total_loss += criterion(logits, labels).item() * features.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


def train(
    data_dir: str | Path,
    model_dir: str | Path,
    config: ModelConfig | None = None,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> Path:
    """Train the VAD model and save a checkpoint after every epoch.

    Args:
        data_dir: Dataset directory containing ``train/`` and ``val/`` splits.
        model_dir: Directory to write checkpoints and TensorBoard logs.
        config: Model architecture. Defaults to ``ModelConfig()`` when omitted.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        device: Compute device. Auto-detected via :func:`get_device` when None.

    Returns:
        Path to the last saved checkpoint.
    """
    torch.manual_seed(0)
    config = config or ModelConfig()
    device = device or get_device()

    data_dir, model_dir = Path(data_dir), Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_loader = create_dataloader(data_dir / "train", batch_size=batch_size, shuffle=True)
    val_loader = create_dataloader(data_dir / "val", batch_size=batch_size, shuffle=False)

    logger.info(f"Starting training on {device} for {epochs} epochs")
    logger.info(
        f"Data: {data_dir} (train: {len(train_loader.dataset)}, "
        f"val: {len(val_loader.dataset)}) | batch_size: {batch_size} | lr: {lr}"
    )
    logger.info(
        f"Model: n_filters={config.n_filters} fc_units={config.fc_units} "
        f"in_channels={config.in_channels} n_classes={config.n_classes}"
    )

    model = Resnet1D(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    writer = SummaryWriter(log_dir=str(model_dir / "logs"))

    last_ckpt = model_dir / f"checkpoint_epoch_{epochs - 1:03d}.pt"
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * features.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()

        train_loss = total_loss / total if total else 0.0
        train_acc = correct / total if total else 0.0
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
        logger.info(
            f"Epoch {epoch}/{epochs - 1} - "
            f"train_loss: {train_loss:.4f} acc: {train_acc:.4f} - "
            f"val_loss: {val_loss:.4f} acc: {val_acc:.4f}"
        )
        writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("acc", {"train": train_acc, "val": val_acc}, epoch)

        last_ckpt = model_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(model.state_dict(), last_ckpt)

    writer.close()
    logger.info(f"Training complete. Checkpoints saved to {model_dir}")
    return last_ckpt


def export_model(
    checkpoint: str | Path,
    out_dir: str | Path,
    config: ModelConfig | None = None,
    torchscript: bool = True,
) -> Path:
    """Export a trained checkpoint for inference.

    Saves the state dict and, optionally, a TorchScript-traced model.

    Args:
        checkpoint: Path to a training checkpoint (state_dict).
        out_dir: Directory to write the exported artifacts.
        config: Model architecture used to rebuild the network. Must match the
            architecture used to train ``checkpoint``. Defaults to
            ``ModelConfig()`` when omitted.
        torchscript: Whether to also export a TorchScript model.

    Returns:
        Path to the exported state-dict file.
    """
    config = config or ModelConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exporting checkpoint {checkpoint} to {out_dir}")

    model = Resnet1D(config)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    state_dict_path = out_dir / "model_state_dict.pt"
    torch.save(model.state_dict(), state_dict_path)
    logger.info(f"Saved state_dict to {state_dict_path}")

    if torchscript:
        dummy_input = torch.randn(1, config.in_channels, FEAT_SIZE[1])
        traced = torch.jit.trace(model, dummy_input)
        script_path = out_dir / "model_scripted.pt"
        traced.save(str(script_path))
        logger.info(f"Saved TorchScript model to {script_path}")

    return state_dict_path


def main() -> None:
    """CLI entrypoint: train the model and (optionally) export it."""
    parser = argparse.ArgumentParser(description="Train a 1D ResNet for Voice Activity Detection.")
    parser.add_argument("--data-dir", "-d", required=True, help="Dataset directory.")
    parser.add_argument("--model-dir", "-m", required=True, help="Checkpoint output directory.")
    parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--batch-size", "-b", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--n-filters", type=int, nargs="+", default=[32, 64, 128, 128], help="Conv filters."
    )
    parser.add_argument(
        "--fc-units", type=int, nargs="+", default=[2048, 2048], help="FC hidden units."
    )
    parser.add_argument("--no-export", action="store_true", help="Skip exporting after training.")
    args = parser.parse_args()

    config = ModelConfig(n_filters=tuple(args.n_filters), fc_units=tuple(args.fc_units))

    last_ckpt = train(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        config=config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    if not args.no_export:
        export_model(
            checkpoint=last_ckpt,
            out_dir=Path(args.model_dir) / "exported",
            config=config,
        )
    else:
        logger.info("Skipping export (--no-export).")


if __name__ == "__main__":
    main()
