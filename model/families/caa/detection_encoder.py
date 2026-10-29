import math
import torch
import torch.nn as nn
from model.common.blocks import Stem, UpSamplingBlock


class Encoder(nn.Module):
    def __init__(self, depth, in_ch, out_ch):
        super().__init__()

        # trainable high-resolution front end (stride 8 -> 32x32), decoder-side
        self.stem = Stem(out_ch=192)

        self.sem = nn.Sequential(
            nn.Conv2d(in_ch * depth, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )  # channel reduction

        self.sem_up = UpSamplingBlock(out_ch, out_ch, ratio=2)  # 16 -> 32 resolution

        self.fuse = nn.Sequential(
            nn.Conv2d((out_ch + 192), out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x, y, xs, ys):

        B = xs[0].shape[0]

        # --- stem: raw images -> 32x32 tokens ---
        hr_x, hr_y = self.stem(x, y)                     # [B, 1024, 192]

        # --- 16x16 -> 32x32 ---
        fx = torch.cat(xs, dim=-1)                       # [B, 256, in_ch*4]
        fy = torch.cat(ys, dim=-1)
        f = torch.cat([fx, fy], dim=0)                   # [2B, 256, in_ch*4]
        N = f.shape[1]
        H = W = math.isqrt(N)                            # 16
        f = f.permute(0, 2, 1).reshape(2 * B, -1, H, W).contiguous()
        f = self.sem(f)                                  # [2B, out_ch, 16, 16]
        f = self.sem_up(f)                               # [2B, out_ch, 32, 32]

        # --- 16x16 -> 32x32 high-resolution fusion ---
        hr = torch.cat([hr_x, hr_y], dim=0)              # [2B, 1024, 192]
        Hr = Wr = math.isqrt(hr.shape[1])                # 32
        hr = hr.permute(0, 2, 1).reshape(2 * B, -1, Hr, Wr).contiguous() # [2B, 192, 32, 32]

        # --- fuse with high-res stem ---
        out = torch.cat([f, hr], dim=1)
        out = self.fuse(out)                              # [2B, out_ch, 32, 32]

        # output contract: flatten to a sequence and chunk into x/y tokens, each [B, 1024, out_ch].
        out = out.flatten(2).transpose(1, 2).contiguous()  # [2B, 1024, out_ch]
        x, y = torch.chunk(out, 2, dim=0)
        return x, y
