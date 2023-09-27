import logging
from functools import partial
from typing import *

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss

import numpy as np

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def log_layer_output(x: torch.Tensor, name: str = ""):
    logger.debug(f"{x.shape} {name}")
    logger.debug(f"{float(x.mean())}, {float(x.std())} {name}")


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
        # logger.debug(f"  {x.shape} INPUT")
        x = self.convlayer1(x)
        # logger.debug(f"  {x.shape} CONVLAYER1")
        res = self.convlayer2(x)
        # logger.debug(f"  {res.shape} CONVLAYER2")
        out = self.pool(res)
        # logger.debug(f"  {out.shape} POOL")
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
        self.convlayer1 = ConvLayer(in_channels, features, scale, **(kwargs1 or {}))
        self.convlayer2 = ConvLayer(2 * features, features, kernel_size, **(kwargs2 or {}))
        self.convlayer3 = ConvLayer(features, features, kernel_size, **(kwargs3 or {}))

    def forward(self, x, res) -> Tuple[torch.Tensor]:
        x = self.upsample(x)
        x = self.convlayer1(x)
        x, res = self.right_crop(x, res)
        x = torch.cat((x, res), dim=1)
        x = self.convlayer2(x)
        return self.convlayer3(x)

    @staticmethod
    def right_crop(x, res) -> Tuple[torch.Tensor, torch.Tensor]:
        x_len, res_len = x.shape[2], res.shape[2]
        if x_len == res_len:
            return x, res

        if x_len < res_len:  # res is bigger, crop res
            logger.debug(">>>> RES IS BIGGER")
        else:  # x is bigger, crop x
            logger.debug("<<<< X IS BIGGER")

        target_len = min(x_len, res_len)
        return x[:, :, :target_len], res[:, :, :target_len]

    @staticmethod
    def center_crop(x, res) -> Tuple[torch.Tensor, torch.Tensor]:
        x_len, res_len = x.shape[2], res.shape[2]
        diff = x_len - res_len
        if diff == 0:
            return x, res

        crop_left = abs(diff) // 2
        crop_right = abs(diff) - crop_left
        if diff < 0:  # res is bigger, crop res
            logger.debug(">>>> RES IS BIGGER")
            return x, res[:, :, crop_left:-crop_right]
        else:  # x is bigger, crop x
            logger.debug("<<<< X IS BIGGER")
            return x[:, :, crop_left:-crop_right], res

# %%

class UTimeBackbone(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        dilation: int = 2,
        kernel_size: int = 5,
        n_features: int = 16,
        complexity_factor: float = 2.0,
        zero_pad: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_channels = n_channels
        self.n_classses = n_classes
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.n_features = n_features
        self.complexity_factor = complexity_factor
        self.zero_pad = zero_pad

        n_filters = int(self.n_features * np.sqrt(complexity_factor))

        self.encoder = nn.ModuleList(
            [
                EncoderBlock(
                    in_channels=n_channels,
                    features=n_filters,
                    kernel_size=kernel_size,
                    pool=10,
                    kwargs1=dict(dilation=dilation, padding="same"),
                    kwargs2=dict(dilation=dilation, padding="same"),
                ),
                EncoderBlock(
                    in_channels=n_filters,
                    features=n_filters * 2,
                    kernel_size=kernel_size,
                    pool=8,
                    kwargs1=dict(dilation=dilation, padding="same"),
                    kwargs2=dict(dilation=dilation, padding="same"),
                ),
                EncoderBlock(
                    in_channels=n_filters * 2,
                    features=n_filters * 4,
                    kernel_size=kernel_size,
                    pool=6,
                    kwargs1=dict(dilation=dilation, padding="same"),
                    kwargs2=dict(dilation=dilation, padding="same"),
                ),
                EncoderBlock(
                    in_channels=n_filters * 4,
                    features=n_filters * 8,
                    kernel_size=kernel_size,
                    pool=4,
                    kwargs1=dict(dilation=dilation, padding="same"),
                    kwargs2=dict(dilation=dilation, padding="same"),
                ),
            ]
        )

        self.bottleneck = nn.Sequential(
            ConvLayer(
                in_channels=n_filters * 8,
                features=n_filters * 16,
                kernel_size=kernel_size,
                padding="same",
            ),
            ConvLayer(
                in_channels=n_filters * 16,
                features=n_filters * 16,
                kernel_size=kernel_size,
                padding="same",
            ),
        )

        self.decoder = nn.ModuleList(
            [
                DecoderBlock(
                    in_channels=n_filters * 16,
                    features=n_filters * 8,
                    kernel_size=kernel_size,
                    scale=4,
                    kwargs1=dict(padding="same"),
                    kwargs2=dict(padding="same"),
                    kwargs3=dict(padding="same"),
                ),
                DecoderBlock(
                    in_channels=n_filters * 8,
                    features=n_filters * 4,
                    kernel_size=kernel_size,
                    scale=6,
                    kwargs1=dict(padding="same"),
                    kwargs2=dict(padding="same"),
                    kwargs3=dict(padding="same"),
                ),
                DecoderBlock(
                    in_channels=n_filters * 4,
                    features=n_filters * 2,
                    kernel_size=kernel_size,
                    scale=8,
                    kwargs1=dict(padding="same"),
                    kwargs2=dict(padding="same"),
                    kwargs3=dict(padding="same"),
                ),
                DecoderBlock(
                    in_channels=n_filters * 2,
                    features=n_filters,
                    kernel_size=kernel_size,
                    scale=10,
                    kwargs1=dict(padding="same"),
                    kwargs2=dict(padding="same"),
                    kwargs3=dict(padding="same"),
                ),
            ]
        )

        self.dense = nn.Sequential(
            nn.Conv1d(
                in_channels=n_filters,
                out_channels=n_classes,
                kernel_size=1,
                padding="same",
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        """A forward pass through the network.

        Args:
            x: Batch of data for network input (batch_size, n_channels, n_timesteps)

        Returns:
            Tensor with shape (batch_size, n_parts, n_classes)
        """
        _n_batches, _n_channels, n_timesteps = x.shape

        logger.debug(f"{x.shape} INPUT")
        residuals = []
        for i, block in enumerate(self.encoder):
            x, residual = block(x)
            logger.debug(f"{residual.shape} RESIDUAL {i}")
            log_layer_output(x, f"ENCODER {i}")
            residuals.append(residual)

        x = self.bottleneck(x)
        log_layer_output(x, "BOTTLENECK")

        for i, (block, residual) in enumerate(zip(self.decoder, residuals[::-1])):
            x = block(x, residual)
            log_layer_output(x, f"DECODER {i}")

        x = self.dense(x)
        log_layer_output(x, "DENSE")

        if self.zero_pad:
            pad_len = n_timesteps - x.shape[-1]
            lpad = pad_len // 2
            rpad = lpad + pad_len % 2
            x = F.pad(x, [lpad, rpad], "constant", 0)

        log_layer_output(x, "OUTPUT")
        return x


class UTimeHead(nn.Module):
    def __init__(
        self,
        n_classes: int,
        pool: Optional[int] = None,
        transition_window: int = 3,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.pool = pool
        self.transition_window = transition_window

        segment_layers = []
        if pool is not None:
            segment_layers.append(nn.AvgPool1d(pool))

        segment_layers.extend(
            [
                nn.Conv1d(
                    in_channels=n_classes,
                    out_channels=n_classes,
                    kernel_size=transition_window,
                    padding="same",
                ),
                nn.Softmax(dim=1),
            ]
        )

        self.segment = nn.Sequential(*segment_layers)

    def forward(self, x):
        x = self.segment(x)
        log_layer_output(x, "SEGMENT")
        return x


class UTime(nn.Sequential):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        backbone_kwargs: Optional[Dict[str, Any]] = None,
        head_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.backbone = UTimeBackbone(
            n_channels,
            n_classes,
            **(backbone_kwargs or {}),
        )
        self.head = UTimeHead(
            n_classes,
            **(head_kwargs or {}),
        )

# %%

model = UTime(
    n_channels=21,
    n_classes=1,
    backbone_kwargs=dict(complexity_factor=1., zero_pad=True),
)

if logger.level <= logging.DEBUG:
    SAMPLE_SECS = 17.5 * 60
    SAMPLE_RATE = 100

    loss_function = (partial(sigmoid_focal_loss, reduction="mean", gamma=2, alpha=0.95),)
    n_timesteps = int(SAMPLE_SECS * SAMPLE_RATE)
    x = torch.rand(3, 21, n_timesteps)
    y = torch.rand(3, n_timesteps, 1)
    try:
        out = model(x)
    except Exception as err:
        print(str(err))
        breakpoint()

