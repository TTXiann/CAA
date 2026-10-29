import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange


class Diff_Gated(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        self.proj = nn.Linear(dim, rank, bias=False)   # shared feature proj W
        self.gate = nn.Linear(dim, rank, bias=False)   # shared gate proj G
        self.alpha = nn.Parameter(torch.zeros(rank))   # interaction scale, zero-init

    def forward(self, s):
        ps = self.proj(s)                                # [2B, N, r]
        gs = torch.sigmoid(self.gate(s))                 # [2B, N, r]
        B = s.shape[0] // 2
        pr = torch.roll(ps, shifts=B, dims=0)            # swap first/second half
        gr = torch.roll(gs, shifts=B, dims=0)
        d = (ps - pr) + self.alpha * (gs * pr - gr * ps)
        return d


class Fusion_Conv(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.block = nn.Conv2d(
            dim, dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False
        )

    def forward(self, x, d):
        N = x.shape[1]
        HW = math.isqrt(N)
        xd = x + d
        x2d = rearrange(xd, 'b (h w) c -> b c h w', h=HW, w=HW).contiguous()
        out = self.block(x2d)
        out = rearrange(out, 'b c h w -> b (h w) c')
        return out


class CAALayer(nn.Module):
    def __init__(self, dim, k, rank):
        super().__init__()
        self.down = nn.Linear(dim, rank * 2, bias=False)
        self.enc = Fusion_Conv(rank, k)
        self.up = nn.Linear(rank, dim, bias=False)
        self.dropout = nn.Dropout(0.1)
        self._reset_parameters()

    def _reset_parameters(self):
        init.zeros_(self.up.weight)

    def forward(self, x, d):
        shortcut = x

        h = self.down(x)

        h1, h2 = torch.chunk(h, 2, dim=-1)
        h = self.enc(h1, d) + h2

        h = F.gelu(h)
        h = self.dropout(h)
        h = self.up(h)

        return h + shortcut


class CAA(nn.Module):
    def __init__(self, dim, rank, k=3):
        super().__init__()
        self.diff = Diff_Gated(dim, rank)
        self.layer = CAALayer(dim, k, rank)

    def forward(self, x, y, num_prefix_tokens):
        x_pref, x_sp = x[:, :num_prefix_tokens], x[:, num_prefix_tokens:]
        y_pref, y_sp = y[:, :num_prefix_tokens], y[:, num_prefix_tokens:]
        sp = torch.cat([x_sp, y_sp], dim=0)   # [2B, N, D]

        sp = self.layer(sp, self.diff(sp))    # [2B, N, D]

        _x_sp, _y_sp = sp.chunk(2, dim=0)
        x_out = torch.cat([x_pref, _x_sp], dim=1)
        y_out = torch.cat([y_pref, _y_sp], dim=1)

        return x_out, y_out


class Adapter_Dual(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.down = nn.Linear(dim, hidden_dim, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(hidden_dim, dim, bias=False)
        self._reset_parameters()

    def _reset_parameters(self):
        init.zeros_(self.up.weight)

    def forward(self, x, y, num_prefix_tokens):
        x_pref, x_sp = x[:, :num_prefix_tokens], x[:, num_prefix_tokens:]
        y_pref, y_sp = y[:, :num_prefix_tokens], y[:, num_prefix_tokens:]

        z = torch.cat([x_sp, y_sp], dim=0)
        residual = z
        z = self.down(z)
        z = self.act(z)
        z = self.up(z)
        z = residual + z

        _x_sp, _y_sp = torch.chunk(z, 2, dim=0)
        x_out = torch.cat([x_pref, _x_sp], dim=1)
        y_out = torch.cat([y_pref, _y_sp], dim=1)
        return x_out, y_out

