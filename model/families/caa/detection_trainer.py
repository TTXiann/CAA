from model.families.caa.backbone import DINOv3_CAA, DINOv2_CAA, CLIP_CAA, MAE_CAA
from model.cd.decoder import MaskPredictor
from model.common.base_trainer import BaseModel
from model.common.utils import weight_init


class CDModel(BaseModel):
    """Change detection model: decoder(x, y, *backbone(x, y)); the variant is picked from --model_name."""

    _freeze_root = 'backbone'

    # variant table: name -> backbone class (the HR stem lives in the decoder)
    VARIANTS = {
        'dinov3caa': DINOv3_CAA,
        'dinov2caa': DINOv2_CAA,
        'clipcaa':   CLIP_CAA,
        'maecaa':    MAE_CAA,
    }

    def __init__(self, args):
        model_name, version = args.model_name.split('_')
        super().__init__()
        if model_name not in self.VARIANTS:
            raise ValueError(f'model_name is not supported: {args.model_name}')
        bb_cls = self.VARIANTS[model_name]
        rank = getattr(args, 'rank', 1)
        depth = getattr(args, 'depth', 1)
        use_caa = getattr(args, 'use_caa', 1)
        self.backbone = bb_cls(version, rank, depth=depth, use_caa=use_caa)

        ch = self.backbone.dim
        self.decoder = MaskPredictor(depth, ch)
        self.decoder.apply(weight_init)

    def forward(self, x, y):
        return self.decoder(x, y, *self.backbone(x, y))
