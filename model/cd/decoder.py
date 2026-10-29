import math

import torch.nn as nn

from model.common.blocks import UpSamplingBlock
from model.families.caa.detection_encoder import Encoder


def _tokens_to_map(out):
    """[B, N, D] sequence -> [B, D, H, W] feature map, N = H*W."""
    B, N, D = out.shape
    H = W = math.isqrt(N)
    return out.permute(0, 2, 1).reshape(B, D, H, W).contiguous()


class MaskPredictor(nn.Module):
    def __init__(self, depth, ch):
        super().__init__()

        # 1) front change encoder (registered first, keep order change_encoder -> up* -> head)
        out_ch = max(ch // 2, 384)
        self.change_encoder = Encoder(depth, ch, out_ch)

        # 2) upsampling tower
        up_chs = [out_ch, 288, 192, 96]
        for i in range(len(up_chs) - 1):
            setattr(self, f'up{i}', UpSamplingBlock(up_chs[i], up_chs[i + 1], ratio=1))
        self._n_up = len(up_chs) - 1

        # 3) classification head (bare 3x3 conv, parameter key head.weight)
        self.head = nn.Conv2d(up_chs[-1], 1, kernel_size=3, stride=1, padding=1, bias=False)

    def run_tower(self, feat):
        """Stage-wise PixelShuffle upsampling."""
        for i in range(self._n_up):
            feat = getattr(self, f'up{i}')(feat)
        return feat

    def forward(self, *args):
        x, y = self.change_encoder(*args)  # [B, N, D]
        out = x + y                        # bi-temporal change fusion (no learnable parameters)
        out = _tokens_to_map(out)
        out = self.run_tower(out)
        return self.head(out)
