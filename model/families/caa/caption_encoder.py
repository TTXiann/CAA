import torch
import torch.nn as nn


class ChangeEncoder(nn.Module):
    def __init__(self, depth, ori_dim, dim, dropout):
        super().__init__()
        if depth != 1:
            raise ValueError(
                f'caption encoder supports n_enc=1 only, got n_enc={depth}')

        self.mlp = nn.Sequential(
            nn.Linear(ori_dim * depth * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.RMSNorm(dim),
        )

    def forward(self, xs, ys):
        out = torch.cat([xs[-1], ys[-1]], dim=-1)   # [B, N, 2*ori_dim]
        return self.mlp(out)                        # [B, N, dim]
