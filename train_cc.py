import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Each process keeps only its own GPU so every rank uses cuda:0 with device_ids=[0].
_local_rank = int(os.environ.get('LOCAL_RANK', 0))
_orig_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if _orig_visible:
    _devices = [d.strip() for d in _orig_visible.split(',')]
    if _local_rank < len(_devices):
        os.environ['CUDA_VISIBLE_DEVICES'] = _devices[_local_rank]

import argparse
import json

import torch
from torch.amp import autocast

from dataset.cc.dataset import CaptionDataset, cc_dataset_key, cc_wordmap_path
from model.common.paths import (DEFAULT_DATA_FOLDER, DEFAULT_SAVE_DIR,
                                cc_dataset_tag, cc_exp_dir, model_version)
from model.common.runner import Runner
from model.trainer import CaptionModel


class CCRunner(Runner):
    task = 'CC'
    metric_name = 'Bleu_4'

    def prepare(self, args):
        args.dataset = cc_dataset_key(args.dataset)
        with open(cc_wordmap_path(args.data_folder, args.dataset), 'r') as f:
            args.vocab = json.load(f)

    def version(self, args):
        return model_version(args.backbone)

    def experiment_dir(self, args):
        return cc_exp_dir(args.save_root, args.dataset, args.exp_dir,
                          args.rank, args.exp_name)

    def describe(self, args):
        return (f'dataset={cc_dataset_tag(args.dataset)} backbone={args.backbone} '
                f'dim={args.dim} n_enc={args.n_enc} n_dec={args.n_dec} '
                f'max_len={args.max_len}')

    def build_model(self, args):
        return CaptionModel(args)

    def build_datasets(self, args):
        return CaptionDataset(args, 'TRAIN'), CaptionDataset(args, args.split)

    def train_step(self, args, batch, model):
        images1, images2, captions = batch
        images1 = images1.cuda(non_blocking=True)
        images2 = images2.cuda(non_blocking=True)
        captions = captions.cuda(non_blocking=True)

        with autocast('cuda', dtype=torch.bfloat16):
            loss = model(images1, images2, captions)

        return loss

    def evaluate(self, args, model, loader):
        from eval_cc import evaluate_transformer
        return evaluate_transformer(
            args=args,
            device=torch.device('cuda'),
            model=model,
            vocab=args.vocab,
            dataloader=loader,
            dist_eval=True,
        )

    def score(self, metrics):
        return metrics['Bleu_4']

    def extra(self, metrics, best):
        s_star = (metrics['Bleu_4'] + metrics['METEOR']
                  + metrics['ROUGE_L'] + metrics['CIDEr']) / 4
        return {'s_star': round(float(s_star), 6)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Change Captioning')

    # data
    parser.add_argument('--data_folder', type=str, default=DEFAULT_DATA_FOLDER,
                        help='dataset root containing RSICD/ and RSICC/')
    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset alias: levir | dubai')
    parser.add_argument('--save_dir', type=str, default=DEFAULT_SAVE_DIR,
                        help='root folder for experiment outputs')
    parser.add_argument('--exp_dir', type=str, default='osnet', help='experiment group folder')

    # model
    parser.add_argument('--backbone', type=str, default='dinov3_small', help='backbone identifier')
    parser.add_argument('--n_enc', type=int, default=1)
    parser.add_argument('--n_dec', type=int, default=1)
    parser.add_argument('--dim', type=int, default=512)
    parser.add_argument('--max_len', type=int, default=41)
    parser.add_argument('--dec_dropout', type=float, default=0.1)

    # CAA adapter
    parser.add_argument('--rank', type=int, default=1, help='CAA bottleneck width')
    parser.add_argument('--use_caa', type=int, default=1,
                        help='1=CAA; 0=Adapter_Dual baseline; -1=no adapter')

    # training
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--eval_batch_size', type=int, default=40)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)

    # evaluation
    parser.add_argument('--split', type=str, default='VAL', choices=['TRAIN', 'VAL', 'TEST'])
    parser.add_argument('--beam_size', type=int, default=3)

    # system
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--local-rank", type=int, default=int(os.environ.get('LOCAL_RANK', 0)))

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    CCRunner(args).fit()
