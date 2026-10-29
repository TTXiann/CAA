import torch.nn as nn


class BaseModel(nn.Module):
    """Training base: only ``caa``/``trainable_*`` modules stay unfrozen and in train mode."""

    _freeze_root = 'encoder'

    def set_finetune(self):
        for n, p in getattr(self, self._freeze_root).named_parameters():
            p.requires_grad = 'caa' in n or 'trainable' in n

    def train(self, mode: bool = True):
        super().train(mode)
        for name, module in getattr(self, self._freeze_root).named_modules():
            if 'caa' in name or 'trainable' in name:
                module.train(mode)
            else:
                module.eval()
        return self
