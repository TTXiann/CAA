import torch
import torch.nn as nn


class UpSamplingBlock(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor=2, ratio=1):
        super().__init__()

        self.conv = nn.Conv2d(
            in_ch,
            out_ch * scale_factor * scale_factor // ratio,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.act_block = nn.Sequential(
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(
                out_ch // ratio,
                out_ch,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

        self._icnr_init(self.conv, upscale_factor=scale_factor)

    def _icnr_init(self, tensor, upscale_factor, init=nn.init.kaiming_normal_):
        out_channels, in_channels, height, width = tensor.weight.shape
        out_channels_sub = out_channels // (upscale_factor ** 2)
        kernel = torch.zeros(out_channels_sub, in_channels, height, width)
        kernel = init(kernel)
        kernel = kernel.repeat(upscale_factor ** 2, 1, 1, 1)
        tensor.weight.data.copy_(kernel)

    def forward(self, x):
        x = self.conv(x)
        x = self.act_block(x)
        return x


class Stem(nn.Module):
    """Trainable 1/8-resolution front end: A/B share one set of convs, 256 -> 32x32 tokens."""

    def __init__(self, out_ch):
        super().__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, out_ch // 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch // 4),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(out_ch // 4, out_ch // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch // 2),
            nn.GELU(),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(out_ch // 2, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x, y):
        h = torch.cat([x, y], dim=0)
        h = self.stage1(h)  # [2B, C_out, H//2, W//2]
        h = self.stage2(h)  # [2B, C_out, H//4, W//4]
        h = self.stage3(h)  # [2B, C_out, H//8, W//8]

        h = h.flatten(2).transpose(1, 2)
        x, y = torch.chunk(h, 2, dim=0)
        return x, y
