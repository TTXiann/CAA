import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.common.utils import weight_init
from model.families.caa.caa import CAA, Adapter_Dual


# Pretrained weight cache dir: default /project/datasets/model_cache, override with CD_MODEL_CACHE
_CACHE = os.environ.get('CD_MODEL_CACHE', '/project/datasets/model_cache')


def _normalise_use_caa(use_caa):
    """1 = CAA, 0 = Adapter_Dual, -1 = no adapter; None keeps the historical default (CAA)."""
    return 1 if use_caa is None else use_caa


class DINOv3_CAA(nn.Module):
    def __init__(self, version, rank, depth, use_caa=None):
        super().__init__()
        use_caa = _normalise_use_caa(use_caa)
        print('DINO V3: ', version)
        from model.backbones.dinov3.vision_transformer import vit_base, vit_small, vit_large

        if 'small' == version:
           self.model = vit_small(
               drop_path_rate=0.0,
               layerscale_init=1.0e-5,
               ffn_layer='mlp',
               mask_k_bias=True,
               n_storage_tokens=4,
               pos_embed_rope_base=100.0,
               pos_embed_rope_normalize_coords='separate',
            )
           msg = self.model.load_state_dict(
               torch.load(f'{_CACHE}/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
                          map_location='cpu'), strict=True)
           self.dim = 384
           if depth == 4:
               self.hook_layers = [2, 5, 8, 11]
           elif depth == 3:
               self.hook_layers = [3, 7, 11]
           elif depth == 2:
               self.hook_layers = [5, 11]
           elif depth == 1:
               self.hook_layers = [11]
           elif depth == 6:
               self.hook_layers = [1, 3, 5, 7, 9, 11]
           else:
               raise ValueError(f'Unsupported depth: {depth}')

        elif 'base' == version:
           self.model = vit_base(
               drop_path_rate=0.0,
               layerscale_init=1.0e-5,
               ffn_layer='mlp',
               mask_k_bias=True,
               n_storage_tokens=4,
               pos_embed_rope_base=100.0,
               pos_embed_rope_normalize_coords='separate',
            )
           msg = self.model.load_state_dict(torch.load(
               f'{_CACHE}/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
               map_location='cpu'), strict=True)
           self.dim = 768
           if depth == 4:
               self.hook_layers = [2, 5, 8, 11]
           elif depth == 3:
               self.hook_layers = [3, 7, 11]
           elif depth == 2:
               self.hook_layers = [5, 11]
           elif depth == 1:
               self.hook_layers = [11]
           else:
               self.hook_layers = [3, 7, 11]

        elif 'large' == version:
           self.model = vit_large(
               drop_path_rate=0.0,
               layerscale_init=1.0e-5,
               ffn_layer='mlp',
               mask_k_bias=True,
               n_storage_tokens=4,
               pos_embed_rope_base=100.0,
               pos_embed_rope_normalize_coords='separate',
            )
           msg = self.model.load_state_dict(torch.load(
               f'{_CACHE}/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
               map_location='cpu'), strict=True)
           self.dim = 1024
           if depth == 4:
               self.hook_layers = [5, 11, 17, 23]
           elif depth == 3:
               self.hook_layers = [7, 15, 23]
           elif depth == 2:
               self.hook_layers = [11, 23]
           elif depth == 1:
               self.hook_layers = [23]
           else:
               self.hook_layers = [7, 15, 23]

        print('backbone load', msg)
        print('hook_layers:', self.hook_layers)

        self.use_adapter = use_caa >= 0
        self.caa = nn.ModuleList([
            CAA(self.dim, rank) if use_caa == 1 else Adapter_Dual(self.dim, rank)
            for _ in range(len(self.model.blocks))
        ]) if self.use_adapter else nn.ModuleList()

        self.num_prefix_tokens = 5

        self.caa.apply(weight_init)

    def forward(self, x, y):
        x, rope = self.model.prepare_tokens_with_masks(x, None)
        y, rope = self.model.prepare_tokens_with_masks(y, None)
        rope_sincos = self.model.rope_embed(H=rope[0], W=rope[1])

        out_x_list = []
        out_y_list = []
        for i, layer in enumerate(self.model.blocks):
            xy = torch.cat([x, y], dim=0)
            xy = layer(xy, rope_sincos)
            x, y = xy.chunk(2, dim=0)

            if self.use_adapter:
                x, y = self.caa[i](x, y, num_prefix_tokens=self.num_prefix_tokens)

            if i in self.hook_layers:
                out_x_list.append(x[:, self.num_prefix_tokens:])
                out_y_list.append(y[:, self.num_prefix_tokens:])

        return out_x_list, out_y_list


class DINOv2_CAA(nn.Module):
    def __init__(self, version, rank, depth, use_caa=None):
        super().__init__()
        use_caa = _normalise_use_caa(use_caa)
        print('DINO V2 CAA: ', version)
        from model.backbones.dinov2.vision_transformer import vit_small, vit_base, vit_large

        if 'small' == version:
            self.model = vit_small(patch_size=14, img_size=518, block_chunks=0, init_values=1.0e-5)
            ckpt = f'{_CACHE}/dinov2_vits14_pretrain.pth'
            self.dim = 384
            if depth == 4:
                self.hook_layers = [2, 5, 8, 11]
            elif depth == 3:
                self.hook_layers = [3, 7, 11]
            elif depth == 2:
                self.hook_layers = [5, 11]
            elif depth == 1:
                self.hook_layers = [11]
            elif depth == 6:
                self.hook_layers = [1, 3, 5, 7, 9, 11]
            else:
                raise ValueError(f'Unsupported depth: {depth}')

        elif 'base' == version:
            self.model = vit_base(patch_size=14, img_size=518, block_chunks=0, init_values=1.0e-5)
            ckpt = f'{_CACHE}/dinov2_vitb14_pretrain.pth'
            self.dim = 768
            if depth == 4:
                self.hook_layers = [2, 5, 8, 11]
            elif depth == 3:
                self.hook_layers = [3, 7, 11]
            elif depth == 2:
                self.hook_layers = [5, 11]
            elif depth == 1:
                self.hook_layers = [11]
            elif depth == 6:
                self.hook_layers = [1, 3, 5, 7, 9, 11]
            else:
                raise ValueError(f'Unsupported depth: {depth}')

        elif 'large' == version:
            self.model = vit_large(patch_size=14, img_size=518, block_chunks=0, init_values=1.0e-5)
            ckpt = f'{_CACHE}/dinov2_vitl14_pretrain.pth'
            self.dim = 1024
            if depth == 4:
                self.hook_layers = [5, 11, 17, 23]
            elif depth == 3:
                self.hook_layers = [7, 15, 23]
            elif depth == 2:
                self.hook_layers = [11, 23]
            elif depth == 1:
                self.hook_layers = [23]
            elif depth == 6:
                self.hook_layers = [3, 7, 11, 15, 19, 23]
            else:
                raise ValueError(f'Unsupported depth: {depth}')

        else:
            raise ValueError(f'Unsupported DINOv2 version: {version}')

        msg = self.model.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=True)
        print('backbone load', msg)
        print('DINOv2 backbone:', version)
        print('hook_layers:', self.hook_layers)

        self.use_adapter = use_caa >= 0
        self.caa = nn.ModuleList([
            CAA(self.dim, rank) if use_caa == 1 else Adapter_Dual(self.dim, rank)
            for _ in range(len(self.model.blocks))
        ]) if self.use_adapter else nn.ModuleList()

        self.num_prefix_tokens = 1

        self.caa.apply(weight_init)

    def _downsample_tokens(self, tokens, src_h=18, src_w=18, dst_h=16, dst_w=16):
        # tokens: [..., src_h*src_w, D] -> [..., dst_h*dst_w, D]
        tokens = rearrange(tokens, '... (h w) d -> ... d h w', h=src_h, w=src_w)
        *leading, d, h, w = tokens.shape
        tokens = tokens.reshape(-1, d, h, w)
        tokens = F.interpolate(tokens, size=(dst_h, dst_w), mode='bilinear', align_corners=False)
        tokens = tokens.reshape(*leading, d, dst_h, dst_w)
        tokens = rearrange(tokens, '... d h w -> ... (h w) d')
        return tokens

    def forward(self, x, y):
        # DINOv2 patch_size=14: resize to 252x252 -> 18x18 tokens, downsampled to 16x16.
        if x.shape[-2:] != (252, 252):
            x = F.interpolate(x, size=(252, 252), mode='bilinear', align_corners=False)
        if y.shape[-2:] != (252, 252):
            y = F.interpolate(y, size=(252, 252), mode='bilinear', align_corners=False)

        x = self.model.prepare_tokens_with_masks(x, None)
        y = self.model.prepare_tokens_with_masks(y, None)

        out_x_list = []
        out_y_list = []

        for i, layer in enumerate(self.model.blocks):
            xy = torch.cat([x, y], dim=0)
            xy = layer(xy)
            x, y = xy.chunk(2, dim=0)

            if self.use_adapter:
                x, y = self.caa[i](x, y, num_prefix_tokens=self.num_prefix_tokens)

            if i in self.hook_layers:
                out_x_list.append(self._downsample_tokens(x[:, self.num_prefix_tokens:]))
                out_y_list.append(self._downsample_tokens(y[:, self.num_prefix_tokens:]))

        return out_x_list, out_y_list


class CLIP_CAA(nn.Module):
    def __init__(self, version, rank, depth, use_caa=None):
        super().__init__()
        use_caa = _normalise_use_caa(use_caa)

        from model.backbones.clip.model import build_model

        if 'base' in version:
            ckpt = torch.load(f'{_CACHE}/ViT-B-16.pt', weights_only=False, map_location='cpu')
            model = build_model(ckpt.state_dict(), img_size=256)
            self.dim = 768
            num_blocks = 12
            if depth == 4:
                self.hook_layers = [2, 5, 8, 11]
            elif depth == 3:
                self.hook_layers = [3, 7, 11]
            elif depth == 2:
                self.hook_layers = [5, 11]
            elif depth == 1:
                self.hook_layers = [11]
            else:
                self.hook_layers = [3, 7, 11]

        else:
            raise ValueError("version is not supported")

        self.model = model.visual.float()
        del model

        print('CLIP backbone:', version)
        print('hook_layers:', self.hook_layers)

        self.num_prefix_tokens = 1

        self.use_adapter = use_caa >= 0
        self.caa = nn.ModuleList([
            CAA(self.dim, rank) if use_caa == 1 else Adapter_Dual(self.dim, rank)
            for _ in range(num_blocks)
        ]) if self.use_adapter else nn.ModuleList()

        self.caa.apply(weight_init)

    def forward(self, x, y):
        # CLIP embedding returns [N, B, C] (LND format)
        x = self.model.embedding(x)
        y = self.model.embedding(y)

        out_x_list = []
        out_y_list = []
        for i, layer in enumerate(self.model.transformer.resblocks):
            # batch x/y through CLIP transformer block (post-norm: Transformer -> CAA)
            xy = torch.cat([x, y], dim=1)      # [N, 2B, C]
            xy = layer(xy)
            x, y = xy.chunk(2, dim=1)          # each [N, B, C]

            # CAA expects [B, N, C]: permute from [N, B, C]
            x = x.permute(1, 0, 2)
            y = y.permute(1, 0, 2)

            if self.use_adapter:
                x, y = self.caa[i](x, y, num_prefix_tokens=self.num_prefix_tokens)

            # hook multi-scale outputs (keep [B, N, C] format, matching DINOv3)
            if i in self.hook_layers:
                out_x_list.append(x[:, self.num_prefix_tokens:])
                out_y_list.append(y[:, self.num_prefix_tokens:])

            # permute back to [N, B, C] for next iteration
            x = x.permute(1, 0, 2)
            y = y.permute(1, 0, 2)

        return out_x_list, out_y_list


class MAE_CAA(nn.Module):
    def __init__(self, version, rank, depth, use_caa=None):
        super().__init__()
        use_caa = _normalise_use_caa(use_caa)

        from model.backbones.mae.models_mae import mae_vit_base_patch16_dec512d8b, mae_vit_large_patch16_dec512d8b
        from model.backbones.mae.util.pos_embed import interpolate_pos_embed

        if 'base' in version:
            self.model = mae_vit_base_patch16_dec512d8b(img_size=256)
            ckpt = torch.load(f'{_CACHE}/mae_pretrain_vit_base.pth', map_location='cpu')['model']
            encoder_ckpt = {k: v for k, v in ckpt.items() if not k.startswith('decoder')}
            interpolate_pos_embed(self.model, encoder_ckpt)  # 256 != pretrain 224
            msg = self.model.load_state_dict(encoder_ckpt, strict=False)
            self.dim = 768
            num_blocks = 12
            if depth == 4:
                self.hook_layers = [2, 5, 8, 11]
            elif depth == 3:
                self.hook_layers = [3, 7, 11]
            elif depth == 2:
                self.hook_layers = [5, 11]
            elif depth == 1:
                self.hook_layers = [11]
            else:
                self.hook_layers = [3, 7, 11]
            # delete decoder modules to save memory
            del self.model.decoder_embed
            del self.model.mask_token
            del self.model.decoder_pos_embed
            del self.model.decoder_blocks
            del self.model.decoder_norm
            del self.model.decoder_pred

        elif 'large' in version:
            self.model = mae_vit_large_patch16_dec512d8b(img_size=256)
            ckpt = torch.load(f'{_CACHE}/mae_pretrain_vit_large.pth', map_location='cpu')['model']
            encoder_ckpt = {k: v for k, v in ckpt.items() if not k.startswith('decoder')}
            interpolate_pos_embed(self.model, encoder_ckpt)  # 256 != pretrain 224
            msg = self.model.load_state_dict(encoder_ckpt, strict=False)
            self.dim = 1024
            num_blocks = 24
            if depth == 4:
                self.hook_layers = [5, 11, 17, 23]
            elif depth == 3:
                self.hook_layers = [7, 15, 23]
            elif depth == 2:
                self.hook_layers = [11, 23]
            elif depth == 1:
                self.hook_layers = [23]
            else:
                self.hook_layers = [7, 15, 23]
                
            del self.model.decoder_embed
            del self.model.mask_token
            del self.model.decoder_pos_embed
            del self.model.decoder_blocks
            del self.model.decoder_norm
            del self.model.decoder_pred

        else:
            raise ValueError("version is not supported")

        print('MAE backbone:', version)
        print('backbone load', msg)
        print('hook_layers:', self.hook_layers)

        self.num_prefix_tokens = 1

        self.use_adapter = use_caa >= 0
        self.caa = nn.ModuleList([
            CAA(self.dim, rank) if use_caa == 1 else Adapter_Dual(self.dim, rank)
            for _ in range(num_blocks)
        ]) if self.use_adapter else nn.ModuleList()

        self.caa.apply(weight_init)

    def forward(self, x, y):
        x = self.model.embedding(x)  # [B, N, C]
        y = self.model.embedding(y)

        out_x_list = []
        out_y_list = []
        for i, layer in enumerate(self.model.blocks):
            # batch x/y through ViT block (post-norm: Transformer -> CAA)
            xy = torch.cat([x, y], dim=0)
            xy = layer(xy)
            x, y = xy.chunk(2, dim=0)

            if self.use_adapter:
                x, y = self.caa[i](x, y, num_prefix_tokens=self.num_prefix_tokens)

            if i in self.hook_layers:
                out_x_list.append(x[:, self.num_prefix_tokens:])
                out_y_list.append(y[:, self.num_prefix_tokens:])

        return out_x_list, out_y_list
