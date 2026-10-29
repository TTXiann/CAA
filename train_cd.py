import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Each process keeps only its own GPU so every rank uses cuda:0 with device_ids=[0].
_local_rank = int(os.environ.get("LOCAL_RANK", 0))
_orig_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if _orig_visible:
    _devices = [d.strip() for d in _orig_visible.split(",")]
    if _local_rank < len(_devices):
        os.environ["CUDA_VISIBLE_DEVICES"] = _devices[_local_rank]

from argparse import ArgumentParser

import torch
from torch.amp import autocast

from dataset.cd import transforms
from dataset.cd.dataset import Dataset, canonical_dataset_alias, resolve_dataset_folder
from model.common.paths import (DEFAULT_DATA_FOLDER, DEFAULT_SAVE_DIR,
                                cd_exp_dir, model_version)
from model.common.runner import Runner
from model.common.utils import BCEDiceLoss
from model.trainer import CDModel


class CDRunner(Runner):
    task = 'CD'
    metric_name = 'F1'

    def prepare(self, args):
        self.dataset_alias = canonical_dataset_alias(args.dataset)
        args.dataset = resolve_dataset_folder(self.dataset_alias)

    def version(self, args):
        return model_version(args.model_name)

    def experiment_dir(self, args):
        return cd_exp_dir(args.save_root, self.dataset_alias, args.exp_dir,
                          args.rank, args.exp_name)

    def describe(self, args):
        return (f'dataset={args.dataset} model={args.model_name} '
                f'depth={args.depth} rank={args.rank}')

    def build_model(self, args):
        return CDModel(args)

    def build_datasets(self, args):
        # substring match over the whole --model_name, first hit wins ('clip' before 'dino')
        backbone = next((b for b in ['clip', 'mae', 'dino'] if b in args.model_name), None)

        if backbone == 'dino':
            train_dataset = Dataset(
                dataset=args.dataset,
                mode='train',
                transform=transforms.DINOTransform(train=True),
                root=args.data_folder)
            test_dataset = Dataset(
                dataset=args.dataset,
                mode=args.split.lower(),
                transform=transforms.DINOTransform(),
                root=args.data_folder)
        elif backbone in ('clip', 'mae'):
            train_dataset = Dataset(
                dataset=args.dataset,
                mode='train',
                transform=transforms.CLIPMAETransform(backbone, train=True),
                root=args.data_folder)
            test_dataset = Dataset(
                dataset=args.dataset,
                mode=args.split.lower(),
                transform=transforms.CLIPMAETransform(backbone),
                root=args.data_folder)
        else:
            raise ValueError(f'{args.model_name} Model not supported')

        return train_dataset, test_dataset

    def train_step(self, args, batch, model):
        image1, image2, target, _ = batch
        image1 = image1.cuda(non_blocking=True).float()
        image2 = image2.cuda(non_blocking=True).float()
        target = target.cuda(non_blocking=True).float()  # [B, 1, H, W]

        with autocast('cuda', dtype=torch.bfloat16):
            output = model(image1, image2)  # [B, C, H, W]
            loss = BCEDiceLoss(output, target)

        return loss

    def evaluate(self, args, model, loader):
        from eval_cd import evaluate_cd
        return evaluate_cd(args, model, loader)

    def score(self, metrics):
        return metrics['class_1'].get('F1', 0.0)


if __name__ == '__main__':
    parser = ArgumentParser(description='Change Detection')

    # data
    parser.add_argument('--data_folder', type=str, default=DEFAULT_DATA_FOLDER,
                        help='dataset root containing RSICD/ and RSICC/')
    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset alias: levir | levir+ | whu | sysu | dsifn | clcd')
    parser.add_argument('--save_dir', type=str, default=DEFAULT_SAVE_DIR,
                        help='root folder for experiment outputs')
    parser.add_argument('--exp_dir', type=str, default='osnet', help='experiment group folder')

    # model
    parser.add_argument('--model_name', type=str, required=True, help='backbone identifier')
    parser.add_argument('--depth', type=int, default=1, help='number of encoder hook layers / change-encoder scales')

    # CAA adapter
    parser.add_argument('--rank', type=int, default=1, help='CAA bottleneck width')
    parser.add_argument('--use_caa', type=int, default=1,
                        help='1=CAA; 0=Adapter_Dual baseline; -1=no adapter')

    # training
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)

    # evaluation
    parser.add_argument('--split', type=str, default='VAL', choices=['TRAIN', 'VAL', 'TEST'])

    # system
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    CDRunner(args).fit()
