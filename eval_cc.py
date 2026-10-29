import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Each process sees only its own GPU (same convention as train_cc.py).
_local_rank = int(os.environ.get('LOCAL_RANK', 0))
_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if _visible:
    _devices = [d.strip() for d in _visible.split(',')]
    if _local_rank < len(_devices):
        os.environ['CUDA_VISIBLE_DEVICES'] = _devices[_local_rank]

import json
import argparse

import torch
import torch.distributed as dist
import torch.utils.data as tud
from tqdm import tqdm

from dataset.cc.dataset import (CaptionDataset, cc_dataset_key,
                                        cc_wordmap_path)
from model.cc.utils import compute_precision_recall, get_eval_score
from model.common.eval_runner import EvalRunner, build_eval_loader
from model.common.paths import (DEFAULT_DATA_FOLDER, DEFAULT_SAVE_DIR,
                                resolve_cc_exp_dir)
from model.common.runner import is_dist_env, is_rank0
from model.trainer import CaptionModel

NOCHANGE_SET = {
    'the scene is the same as before',
    'there is no difference',
    'the two scenes seem identical',
    'no change has occurred',
    'almost nothing has changed',
}


# Text helpers

def normalize_text(s):
    return ' '.join(s.strip().lower().split())


# Collect / aggregate

@torch.inference_mode()
def evaluate_collect(args, device, model, vocab, dataloader):
    model.eval()
    beam_size = args.beam_size
    vocab_inv = {v: k for k, v in vocab.items()}

    pairs = []
    labels = []

    pbar = tqdm(dataloader, desc=f'{args.split} EVAL@BEAM {beam_size}', disable=not is_rank0())

    for (images1, images2, allcaps) in pbar:
        images1 = images1.to(device, non_blocking=True)
        images2 = images2.to(device, non_blocking=True)

        outputs, _ = model.decode_beam(images1, images2, beam_size)

        for ii, sent in enumerate(outputs.tolist()):
            refs = [
                [w for w in c if w not in (vocab['<start>'], vocab['<end>'], vocab['<pad>'])]
                for c in allcaps[ii].tolist()
            ]
            hyp = [w for w in sent if w not in (vocab['<start>'], vocab['<end>'], vocab['<pad>'])]

            raw_gt = ' '.join(vocab_inv[w] for w in refs[0]) if refs else ''
            raw_pred = ' '.join(vocab_inv[w] for w in hyp)

            # 0: no change, 1: change
            gt_label = 0 if normalize_text(raw_gt) in NOCHANGE_SET else 1
            pred_label = 0 if normalize_text(raw_pred) in NOCHANGE_SET else 1

            pairs.append((refs, hyp))
            labels.append((gt_label, pred_label))

    return {
        'pairs': pairs,
        'labels': labels,
    }


def _print_metrics(metrics, beam_size):
    if not is_rank0():
        return
    s_star = (metrics['Bleu_4'] + metrics['METEOR'] + metrics['ROUGE_L'] + metrics['CIDEr']) / 4 * 100
    print('======== Final (corpus-level) ========')
    print(
        'beam size {}: BLEU-1 {} BLEU-2 {} BLEU-3 {} BLEU-4 {} METEOR {} '
        'ROUGE_L {} CIDEr {} s* {}'.format(
            beam_size,
            round(metrics['Bleu_1'] * 100, 2),
            round(metrics['Bleu_2'] * 100, 2),
            round(metrics['Bleu_3'] * 100, 2),
            round(metrics['Bleu_4'] * 100, 2),
            round(metrics['METEOR'] * 100, 2),
            round(metrics['ROUGE_L'] * 100, 2),
            round(metrics['CIDEr'] * 100, 2),
            round(s_star, 2),
        )
    )


def _compute_metrics(all_pairs, all_labels):
    gt_caps = [refs for (refs, _hyp) in all_pairs]
    pred_caps = [hyp for (_refs, hyp) in all_pairs]
    text_metrics, _ = get_eval_score(gt_caps, pred_caps)

    gt_labels = [g for (g, _p) in all_labels]
    pred_labels = [p for (_g, p) in all_labels]
    cls_metrics = compute_precision_recall(gt_labels, pred_labels)

    metrics = dict(text_metrics)
    for cls in (0, 1):
        cls_stats = cls_metrics.get(cls, {'precision': 0.0, 'recall': 0.0})
        metrics[f'cls/{cls}_precision'] = cls_stats['precision']
        metrics[f'cls/{cls}_recall'] = cls_stats['recall']

    return metrics


def aggregate_and_score(args, local_payload, dataset_len):
    pairs = local_payload['pairs']
    labels = local_payload['labels']

    if is_dist_env() and args.dist_eval:
        ws = dist.get_world_size()
        gathered_pairs = [None for _ in range(ws)]
        gathered_labels = [None for _ in range(ws)]

        dist.all_gather_object(gathered_pairs, pairs)
        dist.all_gather_object(gathered_labels, labels)

        metrics = None
        if is_rank0():
            all_pairs = [p for chunk in gathered_pairs for p in (chunk or [])][:dataset_len]
            all_labels = [p for chunk in gathered_labels for p in (chunk or [])][:dataset_len]

            metrics = _compute_metrics(all_pairs, all_labels)
            _print_metrics(metrics, args.beam_size)

        container = [metrics]
        dist.broadcast_object_list(container, src=0)
        return container[0]

    metrics = _compute_metrics(pairs, labels)
    _print_metrics(metrics, args.beam_size)
    return metrics


@torch.inference_mode()
def evaluate_transformer(args, device, model, vocab, dataloader=None, dist_eval=True):
    """Evaluate a CaptionModel: beam-search captions + BLEU/METEOR/ROUGE/CIDEr + change P/R."""
    args.dist_eval = int(dist_eval) if isinstance(dist_eval, bool) else dist_eval

    if dataloader is None:
        args.distributed = getattr(args, 'distributed', int(is_dist_env()))
        dataset = CaptionDataset(args, args.split)
        dataloader = build_eval_loader(args, dataset)
    else:
        dataset = getattr(dataloader, 'dataset', None)
        sampler = getattr(dataloader, 'sampler', None)
        if isinstance(sampler, tud.distributed.DistributedSampler):
            sampler.set_epoch(0)

    payload = evaluate_collect(
        args=args,
        device=device,
        model=model,
        vocab=vocab,
        dataloader=dataloader,
    )

    dataset_len = len(dataset) if dataset is not None else len(getattr(dataloader, 'dataset', []))
    return aggregate_and_score(args, payload, dataset_len)


class CCEvalRunner(EvalRunner):
    task = 'CC'
    config_keys = ('n_enc', 'n_dec', 'dim', 'max_len', 'use_caa')

    def prepare(self, args):
        args.dataset = cc_dataset_key(args.dataset)
        with open(cc_wordmap_path(args.data_folder, args.dataset), 'r') as f:
            args.vocab = json.load(f)

    def experiment_dir(self, args):
        return resolve_cc_exp_dir(
            args.save_dir, args.dataset, args.exp_dir, args.rank, args.exp_name,
            args.n_enc, args.n_dec, args.dim, args.max_len)

    def post_overlay(self, args, cfg):
        # older config.json files store the CAA width as "R"
        args.rank = cfg.get('rank', cfg.get('R', args.rank))

    def build_model(self, args):
        return CaptionModel(args)

    def build_dataset(self, args):
        return CaptionDataset(args, args.split)

    def evaluate(self, args, model, loader):
        return evaluate_transformer(
            args=args, device=torch.device('cuda'), model=model, vocab=args.vocab,
            dataloader=loader, dist_eval=args.dist_eval)

    def summary(self, metrics):
        s_star = (metrics['Bleu_4'] + metrics['METEOR']
                  + metrics['ROUGE_L'] + metrics['CIDEr']) / 4
        return [f's*: {s_star:.4f}']


def run(args):
    CCEvalRunner(args).run()


def build_argparser():
    p = argparse.ArgumentParser(description='ICC post-hoc evaluation (DDP-aware)')

    # data / experiment
    p.add_argument('--data_folder', type=str, default=DEFAULT_DATA_FOLDER,
                   help='dataset root containing RSICD/ and RSICC/')
    p.add_argument('--dataset', type=str, required=True,
                   help='dataset alias: levir | dubai')
    p.add_argument('--save_dir', type=str, default=DEFAULT_SAVE_DIR,
                   help='root folder for experiment outputs')
    p.add_argument('--exp_dir', type=str, default='osnet',
                   help='experiment group folder')
    p.add_argument('--exp_name', type=str, required=True,
                   help='run name, bs{batch_size}_{version}_r{rank} (e.g. bs20_small_r24)')

    # model
    p.add_argument('--backbone', type=str, required=True,
                   help='{name}_{version}, e.g. dinov3_small / dinov2_base / clip_base / mae_base')
    p.add_argument('--n_enc', type=int, default=1)
    p.add_argument('--n_dec', type=int, default=1)
    p.add_argument('--dim', type=int, default=512)
    p.add_argument('--max_len', type=int, default=41, help='levir-cc: 41 | dubai-cc: 27')
    p.add_argument('--dec_dropout', type=float, default=0.1)
    p.add_argument('--rank', type=int, default=1, help='CAA bottleneck width (overlaid from config.json)')
    p.add_argument('--use_caa', type=int, default=1,
                   help='1=CAA; 0=Adapter_Dual baseline; -1=no adapter (overlaid from config.json)')

    # eval
    p.add_argument('--split', default='TEST')
    p.add_argument('--batch_size', type=int, default=100)
    p.add_argument('--beam_size', type=int, default=3)

    # parallel & determinism
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--local_rank', '--local-rank', type=int, default=0)

    return p


if __name__ == '__main__':
    parser = build_argparser()
    args, _ = parser.parse_known_args()
    run(args)
