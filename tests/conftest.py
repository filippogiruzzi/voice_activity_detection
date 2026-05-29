"""Shared pytest fixtures for VAD integration tests."""

import pytest
import torch

from vad.model import ModelConfig


@pytest.fixture()
def model_config() -> ModelConfig:
    """Small model architecture for fast testing."""
    return ModelConfig(
        n_filters=(4, 8),
        kernel_sizes=(3, 3, 3),
        fc_units=(16, 16),
        n_classes=1,
        in_channels=16,
    )


@pytest.fixture()
def train_params() -> dict:
    """Small training hyper-parameters for fast testing."""
    return {"batch_size": 4, "epochs": 2, "lr": 1e-3}


@pytest.fixture()
def dummy_dataset(tmp_path):
    """Create a small dummy .pt dataset with train/val/test splits.

    Returns:
        Path to the dataset root directory.
    """
    n_samples, n_features, time_steps = 8, 16, 65
    for split in ("train", "val", "test"):
        split_dir = tmp_path / split
        split_dir.mkdir()
        for i in range(n_samples):
            sample = {
                "features": torch.randn(n_features, time_steps),
                "label": torch.tensor([float(i % 2)]),
            }
            torch.save(sample, split_dir / f"sample_{i:04d}.pt")
    return tmp_path


@pytest.fixture()
def expected_output_shape():
    """Expected model output shape for a batch of 4 with n_classes=1."""
    return (4, 1)
