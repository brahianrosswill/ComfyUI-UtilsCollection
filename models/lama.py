"""Eager Big LaMa FFC generator.

Adapted from advimman/lama commit 786f5936b27fb3dacd2b1ad799e4de968ea697e7,
saicinpainting/training/modules/ffc.py (Apache-2.0).
"""

from __future__ import annotations

import torch
from torch import nn


class FourierUnit(nn.Module):
    def __init__(self, channels: int, operations):
        super().__init__()
        self.conv_layer = operations.Conv2d(
            channels * 2, channels * 2, kernel_size=1, bias=False
        )
        self.bn = operations.BatchNorm2d(channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch = value.shape[0]
        transformed = torch.fft.rfftn(value, dim=(-2, -1), norm="ortho")
        transformed = torch.stack((transformed.real, transformed.imag), dim=-1)
        transformed = transformed.permute(0, 1, 4, 2, 3).contiguous()
        transformed = transformed.view(batch, -1, *transformed.shape[3:])
        transformed = self.relu(self.bn(self.conv_layer(transformed)))
        transformed = transformed.view(
            batch, -1, 2, *transformed.shape[2:]
        ).permute(0, 1, 3, 4, 2).contiguous()
        transformed = torch.complex(transformed[..., 0], transformed[..., 1])
        return torch.fft.irfftn(
            transformed, s=value.shape[-2:], dim=(-2, -1), norm="ortho"
        )


class SpectralTransform(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, operations):
        super().__init__()
        self.downsample = (
            nn.AvgPool2d(kernel_size=2, stride=2) if stride == 2 else nn.Identity()
        )
        hidden_channels = out_channels // 2
        self.conv1 = nn.Sequential(
            operations.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            operations.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnit(hidden_channels, operations)
        self.conv2 = operations.Conv2d(
            hidden_channels, out_channels, kernel_size=1, bias=False
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.conv1(self.downsample(value))
        return self.conv2(value + self.fu(value))


class FFC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        ratio_gin: float,
        ratio_gout: float,
        operations,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("FFC stride must be 1 or 2")
        in_global = int(in_channels * ratio_gin)
        in_local = in_channels - in_global
        out_global = int(out_channels * ratio_gout)
        out_local = out_channels - out_global
        self.ratio_gout = ratio_gout
        self.global_in_num = in_global

        self.convl2l = self._conv_or_identity(
            operations, in_local, out_local, kernel_size, stride, padding, dilation, bias
        )
        self.convl2g = self._conv_or_identity(
            operations, in_local, out_global, kernel_size, stride, padding, dilation, bias
        )
        self.convg2l = self._conv_or_identity(
            operations, in_global, out_local, kernel_size, stride, padding, dilation, bias
        )
        self.convg2g = (
            nn.Identity()
            if in_global == 0 or out_global == 0
            else SpectralTransform(in_global, out_global, stride, operations)
        )

    @staticmethod
    def _conv_or_identity(
        operations,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        bias,
    ):
        if in_channels == 0 or out_channels == 0:
            return nn.Identity()
        return operations.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            bias=bias,
            padding_mode="reflect",
        )

    def forward(self, value):
        local, global_value = value if isinstance(value, tuple) else (value, 0)
        out_local = 0
        out_global = 0
        if self.ratio_gout != 1:
            out_local = self.convl2l(local) + self.convg2l(global_value)
        if self.ratio_gout != 0:
            out_global = self.convl2g(local) + self.convg2g(global_value)
        return out_local, out_global


class FFCBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        ratio_gin: float,
        ratio_gout: float,
        operations,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        super().__init__()
        self.ffc = FFC(
            in_channels,
            out_channels,
            kernel_size,
            ratio_gin,
            ratio_gout,
            operations,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = (
            nn.Identity()
            if ratio_gout == 1
            else operations.BatchNorm2d(out_channels - global_channels)
        )
        self.bn_g = (
            nn.Identity()
            if ratio_gout == 0
            else operations.BatchNorm2d(global_channels)
        )
        self.act_l = nn.Identity() if ratio_gout == 1 else nn.ReLU(inplace=True)
        self.act_g = nn.Identity() if ratio_gout == 0 else nn.ReLU(inplace=True)

    def forward(self, value):
        local, global_value = self.ffc(value)
        return self.act_l(self.bn_l(local)), self.act_g(self.bn_g(global_value))


class FFCResnetBlock(nn.Module):
    def __init__(self, channels: int, operations):
        super().__init__()
        kwargs = {
            "ratio_gin": 0.75,
            "ratio_gout": 0.75,
            "operations": operations,
            "padding": 1,
        }
        self.conv1 = FFCBNAct(channels, channels, 3, **kwargs)
        self.conv2 = FFCBNAct(channels, channels, 3, **kwargs)

    def forward(self, value):
        local, global_value = value if isinstance(value, tuple) else (value, 0)
        identity_local, identity_global = local, global_value
        local, global_value = self.conv1((local, global_value))
        local, global_value = self.conv2((local, global_value))
        return identity_local + local, identity_global + global_value


class ConcatTupleLayer(nn.Module):
    def forward(self, value):
        local, global_value = value
        return local if not torch.is_tensor(global_value) else torch.cat(value, dim=1)


class FFCResNetGenerator(nn.Module):
    """Exact 18-block generator architecture used by both converted checkpoints."""

    def __init__(self, operations):
        super().__init__()
        model: list[nn.Module] = [
            nn.ReflectionPad2d(3),
            FFCBNAct(4, 64, 7, 0.0, 0.0, operations, padding=0),
        ]
        channels = 64
        for index in range(3):
            next_channels = channels * 2
            ratio_gout = 0.75 if index == 2 else 0.0
            model.append(
                FFCBNAct(
                    channels,
                    next_channels,
                    3,
                    0.0,
                    ratio_gout,
                    operations,
                    stride=2,
                    padding=1,
                )
            )
            channels = next_channels
        model.extend(FFCResnetBlock(channels, operations) for _ in range(18))
        model.append(ConcatTupleLayer())
        for _ in range(3):
            next_channels = channels // 2
            model.extend(
                [
                    operations.ConvTranspose2d(
                        channels,
                        next_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    operations.BatchNorm2d(next_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            channels = next_channels
        model.extend(
            [
                nn.ReflectionPad2d(3),
                operations.Conv2d(64, 3, kernel_size=7),
                nn.Sigmoid(),
            ]
        )
        self.model = nn.Sequential(*model)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.model(value)

    def get_dtype(self):
        return next(self.parameters()).dtype
