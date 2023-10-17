import logging
from math import sqrt
from functools import partial
from typing import Dict, Any, Tuple, Optional, Callable, Union

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss


logger = logging.getLogger(__name__)


class ConvBlock(nn.Sequential):
    """Building block for UTime architecture. Comprised of Convolution, Batch norm, and activation."""

    def __init__(
        self,
        in_channels: int,
        features: int,
        kernel_size: int,
        activation: Optional[Callable] = None,
        conv_kwargs: Optional[Dict[str, Any]] = None,
        bn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Initialise the block.

        Args:
            in_channels: Number of channels in for the convolution layer.
            features: Number of features out for the convolution layer and batch norm.
            kernel_size: Length of kernel for convolution layer.
            activation: Callable for activation after batch norm (by default `nn.ReLU`).
            conv_kwargs: Optional kwargs for convolution layer.
            bn_kwargs: Optional kwargs for batch norm layer.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=features,
            kernel_size=kernel_size,
            **(conv_kwargs or {}),
        )
        self.relu1 = activation or nn.ReLU(inplace=True)
        self.norm1 = nn.BatchNorm1d(num_features=features, **(bn_kwargs or {}))


class EncoderBlock(nn.Module):
    """Block for encoder path in UTime architecture.

    Comprised of 2 `ConvBlock`s and a `MaxPool` layer. Returns a residual in addition to the block
    output, which is the output before the max pool is applied.
    """

    def __init__(
        self,
        in_channels: int,
        features: int,
        kernel_size: int,
        pool: Optional[int] = None,
        conv1_kwargs=None,
        conv2_kwargs=None,
    ):
        """Initialise the block.

        Args:
            in_channels: Number of channels in for the first `ConvBlock`.
            features: Number of features out for the first `ConvBlock` and input/output features
                for the seconds `ConvBlock`.
            kernel_size: Length of kernels for both ConvBlocks.
            pool: Size of max pool kernel. If unset, max pooling will be disabled.
            conv1_kwargs: Optional kwargs for convolution layer of first `ConvBlock`.
            conv2_kwargs: Optional kwargs for convolution layer of second `ConvBlock`.
        """
        super().__init__()
        self.convlayer1 = ConvBlock(
            in_channels,
            features,
            kernel_size,
            conv_kwargs=(conv1_kwargs or {}),
        )
        self.convlayer2 = ConvBlock(
            features,
            features,
            kernel_size,
            conv_kwargs=(conv2_kwargs or {}),
        )
        self.pool = nn.MaxPool1d(kernel_size=pool) if pool is not None else None

    def forward(self, x) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """A forward pass through the Block.

        Args:
            x: A tensor to pass through the block.

        Returns:
            Tuple of output and residual, where the residual is the output of the second ConvBlock
            before the MaxPool is applied. If `self.pool` is `None`, only the residual will be
            returned.
        """
        x = self.convlayer1(x)
        res = self.convlayer2(x)
        if self.pool is None:
            return res
        out = self.pool(res)
        return out, res


class DecoderBlock(nn.Module):
    """Block for decoder path in UTime architecture.

    Comprised of a nearest-neighbour upsampler and 3 `ConvBlock`s. The input is upsampled and passed
    through a`ConvBlock` before the residual for the adjacent `EncoderBlock` is cropped and
    concatenated, which is then passed through 2 more `ConvBlock`s.
    """

    def __init__(
        self,
        in_channels: int,
        features: int,
        scale: int,
        kernel_size: int,
        conv1_kwargs=None,
        conv2_kwargs=None,
        conv3_kwargs=None,
    ):
        """Initialise the block.

        Args:
            in_channels: Number of channels in for the first `ConvBlock`.
            features: Number of features out for the first `ConvBlock` and input/output features
                for the second and third `ConvBlock`.
            scale: Size of upsample kernel.
            kernel_size: Length of kernels for both ConvBlocks.
            conv1_kwargs: Optional kwargs for convolution layer of first `ConvBlock`.
            conv2_kwargs: Optional kwargs for convolution layer of second `ConvBlock`.
            conv3_kwargs: Optional kwargs for convolution layer of third `ConvBlock`.
        """
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale)
        self.convlayer1 = ConvBlock(
            in_channels,
            features,
            scale,
            conv_kwargs=(conv1_kwargs or {}),
        )
        self.convlayer2 = ConvBlock(
            2 * features,
            features,
            kernel_size,
            conv_kwargs=(conv2_kwargs or {}),
        )
        self.convlayer3 = ConvBlock(
            features,
            features,
            kernel_size,
            conv_kwargs=(conv3_kwargs or {}),
        )

    def forward(self, x, res) -> torch.Tensor:
        """A forward pass through the block.

        Args:
            x: Tensor input for the block
            res: Adjacent residual to be cropped and concatenated.

        Returns:
            Tensor output from the block.
        """
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


class UTimeBackbone(nn.Module):
    """Backbone for the UTime architecture.

    Comprised of 4 `EncoderBlock`s, a bottleneck (another `EncoderBlock` without max pool), 4
    `DecoderBlock`s, and 1 dense convolution layer. Output can be optionally zero-padded after the
    last convolution layer to match the shape of the input tensor.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        dilation: int = 2,
        kernel_size: int = 5,
        n_features: int = 16,
        complexity_factor: float = 1.0,
        zero_pad_output: bool = True,
        padding: Optional[str] = "same",
    ):
        """Initialise the backbone.

        Args:
            n_channels: Number of channels of input data.
            n_classes: Number of classes to generate predictions for.
            dilation: Amount of dilation to apply to all convolution layer in `EncoderBlock`s.
            kernel_size: Kernel size for all convolution layer in `EncoderBlock`s and
                `DecoderBlock`s.
            n_features: Number of output features for first `EncoderBlock`. Number of features
                increases by a factor of 2 on each decent down the encoder path.
            complexity_factor: Increases `n_features` by square root of an optional factor.
            zero_pad_output: Whether to zero-pad the decoder output after applying the final convolution
                layer.
            padding: Padding argument for all convolution layers used throughout the encoder and
                decoder paths (default "same"). Set to `None` to disable.
        """
        super().__init__()

        self.n_channels = n_channels
        self.n_classses = n_classes
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.n_features = n_features
        self.complexity_factor = complexity_factor
        self.zero_pad_output = zero_pad_output
        self.padding = padding

        n_filters = int(self.n_features * sqrt(complexity_factor))

        (
            en_filters,
            (bn_in_channels, bn_features),
            de_filters,
        ) = self._get_layer_filters(
            n_channels,
            n_features,
            expansions=[2, 2, 2, 2],
            complexity_factor=complexity_factor,
        )

        pools = [10, 8, 6, 4]
        scales = pools[::-1]

        self.encoder = nn.ModuleList(
            [
                EncoderBlock(
                    in_channels=in_channels,
                    features=features,
                    kernel_size=kernel_size,
                    pool=pool,
                    conv1_kwargs=dict(dilation=dilation, padding=padding),
                    conv2_kwargs=dict(dilation=dilation, padding=padding),
                )
                for (in_channels, features), pool in zip(en_filters, pools)
            ]
        )

        self.bottleneck = EncoderBlock(
            in_channels=bn_in_channels,
            features=bn_features,
            kernel_size=kernel_size,
            conv1_kwargs=dict(dilation=dilation, padding=padding),
            conv2_kwargs=dict(dilation=dilation, padding=padding),
        )

        self.decoder = nn.ModuleList(
            [
                DecoderBlock(
                    in_channels=in_channels,
                    features=features,
                    kernel_size=kernel_size,
                    scale=scale,
                    conv1_kwargs=dict(padding=padding),
                    conv2_kwargs=dict(padding=padding),
                    conv3_kwargs=dict(padding=padding),
                )
                for (in_channels, features), scale in zip(de_filters, scales)
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

    @staticmethod
    def _get_layer_filters(n_channels=21, n_features=16, expansions=None, complexity_factor=1.0):
        if expansions is None:
            expansions = [2, 2, 2, 2]

        fs = [int(n_channels), int(n_features)]
        for expansion in expansions:
            fs.append(int(fs[-1] * expansion))

        for i in range(len(fs) - 1):
            fs[i + 1] = int(fs[i + 1] * sqrt(complexity_factor))

        depth = len(expansions)
        en_filts = [(fs[i], fs[i + 1]) for i in range(depth)]
        bn_filts = (fs[-2], fs[-1])
        de_filts = [(fs[-(i + 1)], fs[-(i + 2)]) for i in range(depth)]

        return en_filts, bn_filts, de_filts

    def forward(self, x):
        """A forward pass through the network.

        Args:
            x: Batch of data for network input `(batch_size, n_channels, n_timesteps)`

        Returns:
            Tensor with shape `(batch_size, n_classes, n_timesteps_out)`. If `self.zero_pad_output = True`,
            then `n_timesteps_out == n_timesteps`.
        """
        _n_batches, _n_channels, n_timesteps = x.shape

        logger.debug(f"{x.shape} INPUT")
        residuals = []
        for i, block in enumerate(self.encoder):
            x, residual = block(x)
            logger.debug(f"{residual.shape} RESIDUAL {i}")
            _log_layer_output(x, f"ENCODER {i}")
            residuals.append(residual)

        x = self.bottleneck(x)
        _log_layer_output(x, "BOTTLENECK")

        for i, (block, residual) in enumerate(zip(self.decoder, residuals[::-1])):
            x = block(x, residual)
            _log_layer_output(x, f"DECODER {i}")

        x = self.dense(x)
        _log_layer_output(x, "DENSE")

        if self.zero_pad_output:
            pad_len = n_timesteps - x.shape[-1]
            lpad = pad_len // 2
            rpad = lpad + pad_len % 2
            x = F.pad(x, [lpad, rpad], "constant", 0)

        _log_layer_output(x, "OUTPUT")
        return x


class UTimeHead(nn.Module):
    """Head for the UTime architecture.

    Comprised of an average pool layer, a convolution layer, and a softmax
    """

    def __init__(
        self,
        n_classes: int,
        pool: Optional[int] = 30 * 100,
        transition_window: int = 3,
        activation: Optional[Callable] = nn.Softmax(dim=2),
    ):
        """Initialise the block.

        Args:
            n_classes: Number of classes to predict.
            pool: Kernel size of average pool. Set to `None` to disable pooling.
            transition_window: Kernel size for final convolution layer.
            activation: Callable to use for the final activation layer.
        """
        super().__init__()
        self.n_classes = n_classes
        self.pool = pool
        self.transition_window = transition_window

        segment_layers = []
        if pool is not None:
            self.pool = int(self.pool)
            segment_layers.append(nn.AvgPool1d(self.pool))

        segment_layers.append(
            nn.Conv1d(
                in_channels=n_classes,
                out_channels=n_classes,
                kernel_size=transition_window,
                padding="same",
            )
        )

        if activation is not None:
            segment_layers.append(activation)

        self.segment = nn.Sequential(*segment_layers)

    def forward(self, x):
        x = self.segment(x)
        _log_layer_output(x, "SEGMENT")
        return x


class UTime(nn.Sequential):
    """Implementation of U-Time in PyTorch using 1D blocks.

    Example:
        >>> SAMPLE_SECS = 17.5 * 60
        >>> SAMPLE_RATE = 100
        >>> N_CHANNELS = 21
        >>> model = UTime(
        ...     n_channels=N_CHANNELS,
        ...     n_classes=5,
        ...     backbone_kwargs=dict(complexity_factor=1., zero_pad=True),
        ...     head_kwargs=dict(pool=30 * SAMPLE_RATE),
        ...     loss_fn=nn.CrossEntropyLoss(),
        ...     lr=1e-5,
        ... )
        >>> x = torch.rand(
        ...     BATCH_SIZE,
        ...     N_CHANNELS,
        ...     SAMPLE_SECS * SAMPLE_RATE,
        ... )
        >>> out = model.forward(x)
    )

    For more details, see:
        * Perslev, M., Jensen, M., Darkner, S., Jennum, P. J., & Igel, C. (2019). U-time: A fully
          convolutional network for time series segmentation applied to sleep staging. Advances in
          Neural Information Processing Systems, 32.
        * https://github.com/perslev/U-Time/tree/utime-paper-version
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        backbone_kwargs: Optional[Dict[str, Any]] = None,
        head_kwargs: Optional[Dict[str, Any]] = None,
        **seermodule_kwargs,
    ):
        """Initialise the model.

        Args:
            n_channels: Number of channels of input data.
            n_classes: Number of classes to generate predictions for.
            backbone_kwargs: Optional kwargs to apply to the `UTimeBackbone`.
            head_kwargs: Optional kwargs to apply to the `UTimeHead`.
            seermodule_kwargs: Arguments to apps through to SeerModule initialiser.
        """
        super().__init__(**seermodule_kwargs)
        self.backbone = UTimeBackbone(
            n_channels,
            n_classes,
            **(backbone_kwargs or {}),
        )
        self.head = UTimeHead(
            n_classes,
            **(head_kwargs or {}),
        )

def _log_layer_output(x: torch.Tensor, name: str = "", log_stats: bool = False):
    """Simple logger wrapper to debug layer dimensions"""
    logger.debug(f"{x.shape} {name}")
    if log_stats:
        logger.debug(f"{float(x.mean())}, {float(x.std())} {name}")


if logger.level <= logging.DEBUG:
    SAMPLE_SECS = 17.5 * 60
    SAMPLE_RATE = 100

    model = UTime(
        n_channels=21,
        n_classes=1,
        backbone_kwargs=dict(complexity_factor=1.0, zero_pad=True),
    )

    n_timesteps = int(SAMPLE_SECS * SAMPLE_RATE)
    x = torch.rand(3, 21, n_timesteps)
    y = torch.rand(3, n_timesteps, 1)

    try:
        out = model(x)
    except Exception as err:
        print(str(err))
        breakpoint()
