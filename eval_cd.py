import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Each process sees only its own GPU (same convention as train_cd.py).
_local_rank = int(os.environ.get('LOCAL_RANK', 0))
_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if _visible:
    _devices = [d.strip() for d in _visible.split(',')]
    if _local_rank < len(_devices):
        os.environ['CUDA_VISIBLE_DEVICES'] = _devices[_local_rank]

import time
from argparse import ArgumentParser

import torch
import torch.distributed as dist
from torch.amp import autocast

from dataset.cd import transforms
from dataset.cd.dataset import Dataset, canonical_dataset_alias, resolve_dataset_folder
from eval_func.cd.metrics import ConfuseMatrixMeter, cm2score
from model.common.eval_runner import EvalRunner
from model.common.paths import (DEFAULT_DATA_FOLDER, DEFAULT_SAVE_DIR,
                                resolve_cd_exp_dir)
from model.common.runner import is_dist_env, is_rank0
from model.trainer import CDModel


@torch.no_grad()
def evaluate_cd(args, model, loader):
    """Confusion-matrix F1/IoU/OA/Kappa over a CD loader, all-reduced under DDP."""
    metric = ConfuseMatrixMeter(n_class=2)

    total_batches = len(loader)
    for it, batch_data in enumerate(loader):
        start_time = time.time()
        image1, image2, target, _ = batch_data
        image1 = image1.cuda().float()
        image2 = image2.cuda().float()
        target = target.cuda().float()

        with autocast('cuda', dtype=torch.bfloat16):
            output = model(image1, image2)  # [B, C, H, W]

        pred = (torch.sigmoid(output) > 0.5).long()
        metric.update_cm(pr=pred.cpu().numpy(), gt=target.cpu().numpy())

        if is_rank0():
            print('\r[%d/%d] time: %.3f' % (
                it, total_batches, time.time() - start_time), end='', flush=True)

    cm = metric.sum

    if is_dist_env():
        cm_tensor = torch.tensor(cm, dtype=torch.float32, device='cuda')
        dist.all_reduce(cm_tensor, op=dist.ReduceOp.SUM)
        return cm2score(cm_tensor.cpu().numpy())

    return metric.get_scores()


class CDEvalRunner(EvalRunner):
    task = 'CD'
    config_keys = ('depth', 'rank', 'use_caa')

    def prepare(self, args):
        self.dataset_alias = canonical_dataset_alias(args.dataset)
        args.dataset = resolve_dataset_folder(self.dataset_alias)

    def experiment_dir(self, args):
        return resolve_cd_exp_dir(args.save_dir, self.dataset_alias, args.dataset,
                                  args.exp_dir, args.rank, args.exp_name)

    def build_model(self, args):
        return CDModel(args)

    def build_dataset(self, args):
        # substring match over the whole --backbone, first hit wins ('clip' before 'dino')
        backbone = next((b for b in ['clip', 'mae', 'dino'] if b in args.backbone), None)

        if backbone == 'dino':
            transform = transforms.DINOTransform()
        elif backbone in ('clip', 'mae'):
            transform = transforms.CLIPMAETransform(backbone)
        else:
            raise ValueError(f'{args.backbone} Model not supported')

        return Dataset(
            dataset=args.dataset,
            mode=args.split.lower(),
            root=args.data_folder,
            transform=transform)

    def evaluate(self, args, model, loader):
        return evaluate_cd(args, model, loader)


if __name__ == '__main__':
    parser = ArgumentParser(description='Change Detection evaluation')

    # data / experiment
    parser.add_argument('--data_folder', type=str, default=DEFAULT_DATA_FOLDER,
                        help='dataset root containing RSICD/ and RSICC/')
    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset alias: levir | levir+ | whu | sysu | dsifn | clcd')
    parser.add_argument('--save_dir', type=str, default=DEFAULT_SAVE_DIR,
                        help='root folder for experiment outputs')
    parser.add_argument('--exp_dir', type=str, default='osnet', help='experiment group folder')
    parser.add_argument('--exp_name', type=str, required=True,
                        help='run name, bs{batch_size}_{version}_r{rank} (e.g. bs16_small_r24)')

    # model
    parser.add_argument('--backbone', type=str, required=True, help='backbone identifier')
    parser.add_argument('--depth', type=int, default=1, help='number of encoder hook layers / change-encoder scales')

    # CAA adapter
    parser.add_argument('--rank', type=int, default=1, help='CAA bottleneck width')
    parser.add_argument('--use_caa', type=int, default=1,
                        help='1=CAA; 0=Adapter_Dual baseline; -1=no adapter')

    # eval
    parser.add_argument('--split', type=str, default='TEST', choices=['TRAIN', 'VAL', 'TEST'])
    parser.add_argument('--batch_size', type=int, default=100)

    # system
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument("--local-rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    CDEvalRunner(args).run()
