import torch
import torch.nn.functional as F

from model.cc.decoder import CaptionDecoder
from model.common.base_trainer import BaseModel
from model.families.caa.backbone import CLIP_CAA, DINOv2_CAA, DINOv3_CAA, MAE_CAA
from model.common.utils import weight_init


BACKBONES = {
    'dinov3': DINOv3_CAA,
    'dinov2': DINOv2_CAA,
    'clip': CLIP_CAA,
    'mae': MAE_CAA,
}


class CaptionModel(BaseModel):
    """Image change captioning: decoder(backbone(x, y), caption); the change encoder lives in the decoder."""

    _freeze_root = 'backbone'

    def __init__(self, args):
        super().__init__()
        self.args = args

        model_name, version = args.backbone.split('_')
        if model_name not in BACKBONES:
            raise ValueError(f'backbone is not supported: {args.backbone}')

        self.backbone = BACKBONES[model_name](
            version,
            rank=args.rank,
            depth=args.n_enc,
            use_caa=getattr(args, 'use_caa', 1))

        self.decoder = CaptionDecoder(
            vocab_size=len(args.vocab),
            max_len=args.max_len,
            depth=args.n_dec,
            dim=args.dim,
            mlp_ratio=2,
            dropout=args.dec_dropout,
            n_enc=args.n_enc,
            ori_dim=self.backbone.dim)

        self.decoder.apply(weight_init)

    def visual_encoding(self, x, y):
        return self.decoder.encode(*self.backbone(x, y))

    def forward(self, x, y, seq):
        memory = self.visual_encoding(x, y)
        max_length = seq.size(1)
        seq_mask = torch.tril(
            torch.ones((1, 1, max_length, max_length), dtype=torch.int32, device=seq.device))
        logits = self.decoder(seq, memory, seq_mask, None)   # [B, L, V]
        scores = logits[:, :-1].reshape(-1, logits.shape[-1]).contiguous()
        targets = seq[:, 1:].reshape(-1).contiguous()
        return F.cross_entropy(scores, targets, ignore_index=0)

    def get_logprobs(self, wt, memory, memory_mask):
        return self.decoder(wt, memory, None, memory_mask)

    @staticmethod
    def _expand_state(batch_size, beam_size, cur_beam_size, selected_beam):
        def fn(s):
            shape = [int(sh) for sh in s.shape]
            beam = selected_beam
            for _ in shape[1:]:
                beam = beam.unsqueeze(-1)
            s = torch.gather(
                s.view(*([batch_size, cur_beam_size] + shape[1:])),
                1,
                beam.expand(*([batch_size, beam_size] + shape[1:])),
            )
            s = s.view(*([-1] + shape[1:]))
            return s
        return fn

    @staticmethod
    def select(batch_size, beam_size, candidate_logprob):
        flat = candidate_logprob.view(batch_size, -1)
        selected_logprob, selected_idx = torch.topk(flat, k=beam_size, dim=-1, largest=True, sorted=True)
        return selected_idx, selected_logprob

    def decode_beam(self, image1, image2, beam_size):
        """Beam search with KV cache; returns (token ids [B, L], log-probs [B, L])."""
        start_idx = self.args.vocab['<start>']
        end_idx = self.args.vocab['<end>']
        max_len = self.args.max_len

        batch_size = image1.size(0)
        device = image1.device

        memory = self.visual_encoding(image1, image2)

        seq_logprob = torch.zeros((batch_size, 1, 1), device=device)
        seq_mask = torch.ones((batch_size, beam_size, 1), device=device)

        log_probs = []
        outputs = []

        wt = torch.empty((batch_size,), dtype=torch.long, device=device).fill_(start_idx)
        self.decoder.init_buffer()

        neg_inf = torch.finfo(seq_logprob.dtype).min
        selected_words = None

        with torch.inference_mode():
            for t in range(max_len):
                cur_beam_size = 1 if t == 0 else beam_size

                logits = self.get_logprobs(wt.unsqueeze(dim=-1), memory, None)
                word_logprob = logits.squeeze(dim=1).log_softmax(dim=-1).view(
                    batch_size, cur_beam_size, -1)

                candidate_logprob = seq_logprob + word_logprob

                # finished beams may only keep emitting <end>
                if t > 0:
                    mask = (selected_words.view(batch_size, cur_beam_size) != end_idx).float().unsqueeze(-1)
                    seq_mask = seq_mask * mask

                    old_seq_logprob = torch.full_like(candidate_logprob, neg_inf)
                    end_col = torch.full((batch_size, cur_beam_size, 1), end_idx,
                                         dtype=torch.long, device=device)
                    old_seq_logprob.scatter_(2, end_col, seq_logprob)
                    candidate_logprob = torch.where(seq_mask.bool(), candidate_logprob, old_seq_logprob)

                selected_idx, selected_logprob = self.select(batch_size, beam_size, candidate_logprob)

                vocab_size = candidate_logprob.shape[-1]
                selected_beam = torch.div(selected_idx, vocab_size, rounding_mode='trunc')
                selected_words = selected_idx - selected_beam * vocab_size

                self.decoder.apply_to_states(
                    self._expand_state(batch_size, beam_size, cur_beam_size, selected_beam))

                seq_logprob = selected_logprob.unsqueeze(-1)                 # [B, beam, 1]
                seq_mask = torch.gather(seq_mask, 1, selected_beam.unsqueeze(-1))

                if outputs:
                    outputs = [torch.gather(o, 1, selected_beam.unsqueeze(-1)) for o in outputs]
                outputs.append(selected_words.unsqueeze(-1))

                this_word_logprob = torch.gather(
                    torch.gather(
                        word_logprob, 1,
                        selected_beam.unsqueeze(-1).expand(batch_size, beam_size, word_logprob.shape[-1])
                    ),
                    2,
                    selected_words.unsqueeze(-1)
                )
                if log_probs:
                    log_probs = [
                        torch.gather(o, 1, selected_beam.unsqueeze(-1).expand(batch_size, beam_size, 1))
                        for o in log_probs
                    ]
                log_probs.append(this_word_logprob)

                wt = selected_words.view(-1)

                if seq_mask.sum() == 0:
                    break

        seq_logprob, sort_idxs = torch.sort(seq_logprob, 1, descending=True)

        outputs = torch.cat(outputs, -1)      # [B, beam, L]
        log_probs = torch.cat(log_probs, -1)

        expand_sort = sort_idxs.expand(batch_size, beam_size, outputs.shape[-1])
        outputs = torch.gather(outputs, 1, expand_sort)
        log_probs = torch.gather(log_probs, 1, expand_sort)

        self.decoder.clear_buffer()

        return outputs.contiguous()[:, 0], log_probs.contiguous()[:, 0]
