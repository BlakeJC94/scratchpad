import logging
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss

import numpy as np

logging.basicConfig()
logger = logging.getLogger(__name__)

class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channels,
        features,
        kernel_size,
        activation=None,
        **kwargs,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=features,
            kernel_size=kernel_size,
            **kwargs,
        )
        self.norm1 = nn.BatchNorm1d(num_features=features)
        self.relu1 = activation or nn.ReLU(inplace=True)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        features,
        kernel_size,
        pool,
        kwargs1=None,
        kwargs2=None,
    ):
        super().__init__()
        self.convlayer1 = ConvLayer(
            in_channels,
            features,
            kernel_size,
            **(kwargs1 or {}),
        )
        self.convlayer2 = ConvLayer(
            features,
            features,
            kernel_size,
            **(kwargs2 or {}),
        )
        self.pool = nn.MaxPool1d(kernel_size=pool)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        logger.debug(f"  {x.shape} INPUT")
        x = self.convlayer1(x)
        logger.debug(f"  {x.shape} CONVLAYER1")
        res = self.convlayer2(x)
        logger.debug(f"  {res.shape} CONVLAYER2")
        out = self.pool(res)
        logger.debug(f"  {out.shape} POOL")
        return out, res


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        features,
        scale,
        kernel_size,
        kwargs1=None,
        kwargs2=None,
        kwargs3=None,
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale)
        self.convlayer1 = ConvLayer( in_channels, features, scale, **(kwargs1 or {}))
        self.convlayer2 = ConvLayer( 2 * features, features, kernel_size, **(kwargs2 or {}))
        self.convlayer3 = ConvLayer( features, features, kernel_size, **(kwargs3 or {}))

    def forward(self, x, res) -> Tuple[torch.Tensor]:
        x = self.upsample(x)
        x = self.convlayer1(x)
        x, res = self.center_crop(x, res)
        x = torch.cat((x, res), dim=1)
        x = self.convlayer2(x)
        return self.convlayer3(x)

    @staticmethod
    def center_crop(x, res) -> Tuple[torch.Tensor, torch.Tensor]:
        x_len, res_len = x.shape[2], res.shape[2]
        diff = x_len - res_len
        if diff == 0:
            return x, res

        crop_left = abs(diff) // 2
        crop_right = abs(diff) - crop_left
        if diff < 0:  # res is bigger, crop res
            return x, res[:, :, crop_left:-crop_right]
        else:  # x is bigger, crop x
            return x[:, :, crop_left:-crop_right], res


class UTimeBackbone(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        dilation: int = 2,
        kernel_size: int = 5,
        n_features: int = int(16 * np.sqrt(2)),  # Complexity factor
        transition_window: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_channels = n_channels
        self.n_classses = n_classes
        self.kernel_size = kernel_size
        self.dilation = dilation

        self.encoder = nn.ModuleList(
            [
                EncoderBlock(
                    in_channels=n_channels,
                    features=n_features,
                    kernel_size=kernel_size,
                    pool=10,
                    kwargs1=dict(dilation=dilation, padding="same"),
                    kwargs2=dict(dilation=dilation, padding="same"),
                ),
                EncoderBlock(
                    in_channels=n_features,
                    features=n_features * 2,
                    kernel_size=kernel_size,
                    pool=8,
                    kwargs1=dict(dilation=dilation, padding="same"),
                    kwargs2=dict(dilation=dilation, padding="same"),
                ),
                EncoderBlock(
                    in_channels=n_features * 2,
                    features=n_features * 4,
                    kernel_size=kernel_size,
                    pool=6,
                    kwargs1=dict(dilation=dilation, padding="same"),
                    kwargs2=dict(dilation=dilation, padding="same"),
                ),
                EncoderBlock(
                    in_channels=n_features * 4,
                    features=n_features * 8,
                    kernel_size=kernel_size,
                    pool=4,
                    kwargs1=dict(dilation=dilation, padding="same"),
                    kwargs2=dict(dilation=dilation, padding="same"),
                ),
            ]
        )

        self.bottleneck = nn.Sequential(
            ConvLayer(
                in_channels=n_features * 8,
                features=n_features * 16,
                kernel_size=kernel_size,
                padding="same",
            ),
            ConvLayer(
                in_channels=n_features * 16,
                features=n_features * 16,
                kernel_size=kernel_size,
                padding="same",
            ),
        )

        self.decoder = nn.ModuleList(
            [
                DecoderBlock(
                    in_channels=n_features * 16,
                    features=n_features * 8,
                    kernel_size=kernel_size,
                    scale=4,
                    kwargs1=dict(padding="same"),
                    kwargs2=dict(padding="same"),
                    kwargs3=dict(padding="same"),
                ),
                DecoderBlock(
                    in_channels=n_features * 8,
                    features=n_features * 4,
                    kernel_size=kernel_size,
                    scale=6,
                    kwargs1=dict(padding="same"),
                    kwargs2=dict(padding="same"),
                    kwargs3=dict(padding="same"),
                ),
                DecoderBlock(
                    in_channels=n_features * 4,
                    features=n_features * 2,
                    kernel_size=kernel_size,
                    scale=8,
                    kwargs1=dict(padding="same"),
                    kwargs2=dict(padding="same"),
                    kwargs3=dict(padding="same"),
                ),
                DecoderBlock(
                    in_channels=n_features * 2,
                    features=n_features,
                    kernel_size=kernel_size,
                    scale=10,
                    kwargs1=dict(padding="same"),
                    kwargs2=dict(padding="same"),
                    kwargs3=dict(padding="same"),
                ),
            ]
        )

        head_filters = int(np.sqrt(2) * n_classes)
        self.head = nn.Sequential(
            nn.Conv1d(
                in_channels=n_features,
                out_channels=head_filters,
                kernel_size=1,
                padding="same",
            ),
            nn.Tanh(),
            nn.Conv1d(
                in_channels=head_filters,
                out_channels=n_classes,
                kernel_size=transition_window,
                padding="same",
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=n_classes,
                out_channels=n_classes,
                kernel_size=transition_window,
                padding="same",
            ),
        )

    def forward(self, x):
        """A forward pass through the network.

        Args:
            x: Batch of data for network input (batch_size, n_channels, n_timesteps)

        Returns:
            Tensor with shape (batch_size, n_parts, n_classes)
        """
        logger.debug(f"{x.shape} INPUT")
        residuals = []
        for i, block in enumerate(self.encoder):
            x, residual = block(x)
            logger.debug(f"{residual.shape} RESIDUAL {i}")
            logger.debug(f"{x.shape} ENCODER {i}")
            logger.debug(f"{float(x.mean())}, {float(x.std())} ENCODER {i}")
            residuals.append(residual)

        x = self.bottleneck(x)
        logger.debug(f"{x.shape} BOTTLENECK")
        logger.debug(f"{float(x.mean())}, {float(x.std())} BASE")

        for i, (block, residual) in enumerate(zip(self.decoder, residuals[::-1])):
            x = block(x, residual)
            logger.debug(f"{x.shape} DECODER {i}")
            logger.debug(f"{float(x.mean())}, {float(x.std())} DECODER {i}")

        x = self.head(x)
        logger.debug(f"{x.shape} HEAD")
        logger.debug(f"{float(x.mean())}, {float(x.std())} HEAD")

        x = x.permute(0, 2, 1)
        logger.debug(f"{x.shape} OUTPUT")
        logger.debug(f"{float(x.mean())}, {float(x.std())} OUTPUT")

        return x


class UTime(nn.Sequential):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        dilation: int = 2,
        kernel_size: int = 5,
        n_features: int = int(16 * np.sqrt(2)),  # Complexity factor
        transition_window: int = 3,
        **backbone_kwargs,
    ):
        super().__init__()
        self.backbone = UTimeBackbone(**backbone_kwargs)


model = UTime(
    n_channels=21,
    n_classes=1,
    loss_function=partial(sigmoid_focal_loss, reduction='mean', gamma=2, alpha=0.95),
    lr=1e-3,
)

if logger.level <= logging.DEBUG:
    loss_function=partial(sigmoid_focal_loss, reduction='mean', gamma=2, alpha=0.95),
    n_timesteps = int(SAMPLE_SECS * SAMPLE_RATE)
    x = torch.rand(3, 21, n_timesteps)
    y = torch.rand(3, n_timesteps, 1)
    out = model(x)
    # loss = nn.BCEWithLogitsLoss()(out, y)
    loss = FocalWithLogitsLoss()(out, y)
    breakpoint()

