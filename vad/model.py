"""PyTorch 1D ResNet model for Voice Activity Detection (VAD).

The model consumes stacked acoustic features (MFCC + delta + delta2 + RMS) of
shape ``(batch, in_channels, time)`` and produces one logit per sample that is
interpreted, after a sigmoid, as the probability of the window containing speech.
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    """Immutable architecture hyper-parameters for :class:`Resnet1D`.

    Grouping these values in a single frozen dataclass keeps the model,
    training, export, and inference code in sync and avoids passing the same
    five arguments around by hand.

    Attributes:
        n_filters: Output channel count for each residual block. The length of
            this tuple defines how many residual blocks are stacked.
        kernel_sizes: The three convolution kernel sizes shared by every
            residual block.
        fc_units: Hidden unit counts for the fully connected classification
            head. Only the first two entries are used.
        n_classes: Number of output logits (1 for binary voice activity).
        in_channels: Number of input feature channels (16 for the stacked
            MFCC + delta + delta2 + RMS features).
    """

    n_filters: tuple[int, ...] = (32, 64, 128, 128)
    kernel_sizes: tuple[int, int, int] = (8, 5, 3)
    fc_units: tuple[int, ...] = (2048, 2048)
    n_classes: int = 1
    in_channels: int = 16


class ResnetBlock(nn.Module):
    """1D residual block: three conv layers plus a 1x1 projection shortcut.

    The block applies three ``Conv1d -> BatchNorm1d`` stages (the first two
    followed by ReLU) and adds a batch-normalised 1x1 convolution of the input
    as the residual shortcut before the final ReLU. ``padding="same"`` keeps the
    temporal dimension unchanged, so input and output share the same time axis.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: tuple[int, int, int] = (8, 5, 3),
    ) -> None:
        """Initialize the residual block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (filters) for every conv.
            kernel_sizes: Kernel sizes for the three stacked conv layers.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_sizes[0], padding="same")
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_sizes[1], padding="same")
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_sizes[2], padding="same")
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn_shortcut = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual block.

        Args:
            x: Input tensor of shape ``(batch, in_channels, time)``.

        Returns:
            Output tensor of shape ``(batch, out_channels, time)``.
        """
        identity = self.bn_shortcut(self.shortcut(x))
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class Resnet1D(nn.Module):
    """1D ResNet for VAD: residual blocks -> global pooling -> FC head.

    The network stacks one :class:`ResnetBlock` per entry in
    ``config.n_filters``, collapses the time axis with global average pooling,
    and classifies the pooled features with a three-layer fully connected head.

    Input shape: ``(batch, in_channels, time)``.
    Output shape: ``(batch, n_classes)`` (raw logits).
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        """Initialize the model from a :class:`ModelConfig`.

        Args:
            config: Architecture hyper-parameters. Defaults to ``ModelConfig()``
                (the production-sized network) when omitted.
        """
        super().__init__()
        self.config = config or ModelConfig()

        blocks: list[nn.Module] = []
        channels = self.config.in_channels
        for out_channels in self.config.n_filters:
            blocks.append(ResnetBlock(channels, out_channels, self.config.kernel_sizes))
            channels = out_channels
        self.blocks = nn.Sequential(*blocks)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.config.n_filters[-1], self.config.fc_units[0]),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.fc_units[0], self.config.fc_units[1]),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.fc_units[1], self.config.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Input feature tensor of shape ``(batch, in_channels, time)``.

        Returns:
            Logit tensor of shape ``(batch, n_classes)``.
        """
        x = self.blocks(x)
        # Collapse the time axis: (batch, channels, time) -> (batch, channels).
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)
