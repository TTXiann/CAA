import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.families.caa.caption_encoder import ChangeEncoder


class FFN(nn.Module):
    def __init__(self, dim, mlp_ratio, dropout=0., act=nn.GELU()):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SelfMHA(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.time_step = None
        self.buffer_key = None
        self.buffer_value = None

    def init_buffer(self):
        self.time_step = 0
        self.buffer_key = None
        self.buffer_value = None

    def clear_buffer(self):
        self.buffer_key = None
        self.buffer_value = None
        self.time_step = None

    def apply_to_states(self, fn):
        if self.buffer_key is not None:
            self.buffer_key = fn(self.buffer_key)
        if self.buffer_value is not None:
            self.buffer_value = fn(self.buffer_value)

    def forward(self, q, k, v, mask=None):
        bs, nq = q.shape[:2]
        nk = k.shape[1]

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.view(bs, nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.view(bs, nk, self.num_heads, self.head_dim).permute(0, 2, 3, 1).contiguous()
        v = v.view(bs, nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        if self.time_step is not None:  # AR mode: append the new key/value to the cache
            if self.time_step == 0:
                self.buffer_key = k
                self.buffer_value = v
            else:
                self.buffer_key = torch.cat([self.buffer_key, k], dim=-1)
                self.buffer_value = torch.cat([self.buffer_value, v], dim=2)

            k = self.buffer_key
            v = self.buffer_value
            self.time_step += 1

        attn = torch.matmul(q, k) * self.scale  # [B, H, N_q, N_k]

        if mask is not None:
            mask_bool = mask == 0 if mask.dtype != torch.bool else ~mask
            attn = attn.masked_fill(mask_bool, torch.finfo(attn.dtype).min)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(bs, nq, -1)
        out = self.o_proj(out)
        out = self.dropout(out)
        return out


class CrossMHA(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.time_step = None
        self.buffer_key = None
        self.buffer_value = None

    def init_buffer(self):
        self.time_step = 0
        self.buffer_key = None
        self.buffer_value = None

    def clear_buffer(self):
        self.time_step = None
        self.buffer_key = None
        self.buffer_value = None

    def forward(self, q, k, v, mask=None):
        bs = q.shape[0]

        q = self.q_proj(q)
        q = rearrange(q, 'b n (h c) -> b h n c', h=self.num_heads).contiguous()

        if self.time_step is not None:  # AR mode: cache the encoder memory once
            if self.time_step == 0:
                k_proj = self.k_proj(k)
                v_proj = self.v_proj(v)
                k_proj = rearrange(k_proj, 'b n (h c)-> b h c n', h=self.num_heads).contiguous()
                v_proj = rearrange(v_proj, 'b n (h c)-> b h n c', h=self.num_heads).contiguous()
                self.buffer_key = k_proj
                self.buffer_value = v_proj

            # beam search expands the batch on the first decoding step
            if self.time_step == 1 and bs != self.buffer_key.shape[0]:
                repeat_factor = bs // self.buffer_key.shape[0]
                if repeat_factor > 1:
                    self.buffer_key = self.buffer_key.repeat_interleave(repeat_factor, dim=0)
                    self.buffer_value = self.buffer_value.repeat_interleave(repeat_factor, dim=0)

            k_proj = self.buffer_key
            v_proj = self.buffer_value
            self.time_step += 1

        else:  # training / full-sequence
            k_proj = self.k_proj(k)
            v_proj = self.v_proj(v)
            k_proj = rearrange(k_proj, 'b n (h c)-> b h c n', h=self.num_heads).contiguous()
            v_proj = rearrange(v_proj, 'b n (h c)-> b h n c', h=self.num_heads).contiguous()

        attn = torch.matmul(q, k_proj) * self.scale  # [B, H, N_q, N_k]

        if mask is not None:
            mask_bool = mask == 0 if mask.dtype != torch.bool else ~mask
            attn = attn.masked_fill(mask_bool, torch.finfo(attn.dtype).min)

        attn = F.softmax(attn, dim=-1)
        attn_ = self.dropout(attn)

        out = torch.matmul(attn_, v_proj)
        out = rearrange(out, 'b h n c -> b n (h c)').contiguous()
        out = self.o_proj(out)
        out = self.dropout(out)
        return out


class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim, mlp_ratio, dropout, act):
        super().__init__()
        self.attn = SelfMHA(dim, dim // 64, dropout)
        self.cross_attn = CrossMHA(dim, dim // 64, dropout)
        self.ffn = FFN(dim, mlp_ratio, dropout, act)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def init_buffer(self):
        self.attn.init_buffer()
        self.cross_attn.init_buffer()

    def clear_buffer(self):
        self.attn.clear_buffer()
        self.cross_attn.clear_buffer()

    def apply_to_states(self, fn):
        self.attn.apply_to_states(fn)

    def forward(self, x, memory, x_mask, memory_mask):
        shortcut = x
        x = self.attn(x, x, x, x_mask)
        x = self.norm1(x + shortcut)

        shortcut = x
        x = self.cross_attn(x, memory, memory, memory_mask)
        x = self.norm2(x + shortcut)

        shortcut = x
        x = self.ffn(x)
        x = self.norm3(x + shortcut)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, max_len, depth, dim, mlp_ratio, dropout=0.0, act=nn.GELU()):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(dim, mlp_ratio, dropout, act)
            for _ in range(depth)
        ])

        self.word_embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, dim)
        self.generator = nn.Linear(dim, vocab_size, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.clear_buffer()

    def init_buffer(self):
        """Call once before autoregressive decoding."""
        self.seq_len = 0
        for layer in self.layers:
            layer.init_buffer()

    def clear_buffer(self):
        """Reset to the training / full-sequence mode."""
        self.seq_len = None
        for layer in self.layers:
            layer.clear_buffer()

    def apply_to_states(self, fn):
        for layer in self.layers:
            layer.apply_to_states(fn)

    def forward(self, seq, memory, seq_mask=None, memory_mask=None):
        """seq: [B, L] ids, seq_mask: [1 or B, 1, L, L], memory: [B, R, D]; returns logits [B, L, vocab_size]."""
        seq_len = seq.size(1)

        if self.seq_len is not None:
            # AR mode: may decode one or more tokens per call
            pos_indx = torch.arange(
                self.seq_len, self.seq_len + seq_len, device=seq.device
            ).view(1, -1)
            self.seq_len += seq_len
        else:
            pos_indx = torch.arange(0, seq_len, device=seq.device).view(1, -1)

        x = self.word_embed(seq) + self.pos_embed(pos_indx)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, memory, seq_mask, memory_mask)

        x = self.dropout(x)
        x = self.generator(x)
        return x


class CaptionDecoder(nn.Module):
    """CC decoder: change encoder over backbone features + transformer caption decoder, all under ``decoder``."""

    def __init__(self, vocab_size, max_len, depth, dim, mlp_ratio, dropout=0.0,
                 n_enc=1, ori_dim=384, act=nn.GELU()):
        super().__init__()

        self.change_encoder = ChangeEncoder(n_enc, ori_dim, dim, dropout)
        self.transformer = TransformerDecoder(
            vocab_size=vocab_size,
            max_len=max_len,
            depth=depth,
            dim=dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            act=act)

    def encode(self, xs, ys):
        """Backbone features -> decoder memory [B, N, dim]."""
        return self.change_encoder(xs, ys)

    def forward(self, seq, memory, seq_mask=None, memory_mask=None):
        return self.transformer(seq, memory, seq_mask, memory_mask)

    def init_buffer(self):
        self.transformer.init_buffer()

    def clear_buffer(self):
        self.transformer.clear_buffer()

    def apply_to_states(self, fn):
        self.transformer.apply_to_states(fn)
