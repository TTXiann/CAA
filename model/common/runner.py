import json
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model.common.logger import ExperimentLogger
from model.common.paths import format_exp_name


def configure_env(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def init_seed(seed, rank):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)


def is_dist_env():
    return dist.is_available() and dist.is_initialized()


def is_rank0():
    return not is_dist_env() or dist.get_rank() == 0


def setup_distributed(args):
    """Set the world size, init NCCL and seed every rank."""
    args.distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1

    if args.distributed:
        torch.cuda.set_device(0)
        dist.init_process_group(backend='nccl', init_method='env://',
                                device_id=torch.device('cuda:0'))

    init_seed(args.seed, rank=dist.get_rank() if is_dist_env() else 0)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loaders(args, train_dataset, test_dataset):
    g = torch.Generator()
    g.manual_seed(args.seed + (dist.get_rank() if is_dist_env() else 0))

    train_sampler, test_sampler = None, None
    if args.distributed:
        train_sampler = DistributedSampler(
            train_dataset, shuffle=True, drop_last=True, seed=args.seed)
        test_sampler = DistributedSampler(
            test_dataset, shuffle=False, drop_last=False, seed=args.seed)

    shared = dict(num_workers=args.num_workers, pin_memory=True,
                  worker_init_fn=seed_worker, generator=g,
                  persistent_workers=(args.num_workers > 0))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=train_sampler is None, sampler=train_sampler,
        drop_last=True, **shared)

    test_loader = DataLoader(
        test_dataset, batch_size=args.eval_batch_size,
        shuffle=False, sampler=test_sampler,
        drop_last=False, **shared)

    return train_loader, test_loader


def scale_metrics(metrics, factor=100.0):
    return {
        key: ({k: v * factor for k, v in value.items()} if isinstance(value, dict)
              else value * factor if isinstance(value, (int, float)) and not isinstance(value, bool)
              else value)
        for key, value in metrics.items()
    }


def print_metrics(metrics):
    for key, value in metrics.items():
        if isinstance(value, dict):
            inner = ' '.join(f'{k}: {v:.4f}' for k, v in value.items())
            print(f'{key} -> {inner}')
        elif isinstance(value, float):
            print(f'{key}: {value:.4f}')
        else:
            print(f'{key}: {value}')


class Runner:
    """Shared training pipeline; subclasses only provide task-specific hooks."""

    task = 'task'
    metric_name = 'metric'

    def __init__(self, args):
        self.args = args
        self.logger = None

    # ----- task hooks -----

    def prepare(self, args):
        """Task-level preparation before the model is built."""

    def version(self, args):
        raise NotImplementedError

    def experiment_dir(self, args):
        raise NotImplementedError

    def describe(self, args):
        return ''

    def build_model(self, args):
        raise NotImplementedError

    def build_datasets(self, args):
        raise NotImplementedError

    def train_step(self, args, batch, model):
        raise NotImplementedError

    def evaluate(self, args, model, loader):
        raise NotImplementedError

    def score(self, metrics):
        raise NotImplementedError

    def extra(self, metrics, best):
        return {}

    # ----- shared pipeline -----

    def fit(self):
        args = self.args
        configure_env(args.seed)
        setup_distributed(args)

        self.prepare(args)

        # the run name is always derived from the actual batch size and rank
        args.exp_name = format_exp_name(args.batch_size, self.version(args), args.rank)

        args.save_root = args.save_dir
        args.save_dir = self.experiment_dir(args)

        model = self.build_model(args)
        model.set_finetune()
        model.cuda()

        if args.distributed:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[0], output_device=0,
                        broadcast_buffers=True, find_unused_parameters=False)

        if is_rank0():
            print(model)
            os.makedirs(args.save_dir, exist_ok=True)

            # dump the full config before training for reproducibility
            config = vars(args).copy()
            config.pop('vocab', None)
            with open(os.path.join(args.save_dir, 'config.json'), 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)

            with open(os.path.join(args.save_dir, 'model_print.txt'), 'w') as f:
                print(model, file=f)

            self.logger = ExperimentLogger(args.save_dir)
            self.logger.info(
                f'{self.task} train | exp={os.path.basename(args.save_dir.rstrip("/"))} '
                f'{self.describe(args)} batch={args.batch_size} lr={args.lr} seed={args.seed}')

        train_dataset, test_dataset = self.build_datasets(args)
        if is_rank0():
            print(f'Train dataset size: {len(train_dataset)}, '
                  f'Test dataset size: {len(test_dataset)}')

        train_loader, test_loader = build_loaders(args, train_dataset, test_dataset)

        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train_loader) * args.epoch, eta_min=args.lr * 0.01)

        best = 0.0
        start_time = time.time()

        for epoch in range(1, args.epoch + 1):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            model.train()
            train_loss, train_time = self.train_epoch(
                args, train_loader, model, optimizer, lr_scheduler, epoch)

            if is_dist_env():
                dist.barrier()

            model.eval()
            eval_start = time.time()
            metrics = self.evaluate(
                args, model.module if args.distributed else model, test_loader)
            eval_time = time.time() - eval_start

            if is_dist_env():
                dist.barrier()

            if is_rank0():
                shown = scale_metrics(metrics)
                print_metrics(shown)

                value = float(self.score(metrics))
                is_best = value > best
                best = max(value, best)
                if is_best:
                    self.save(model, 'best_model.pth')
                    print(f'New best {self.metric_name}: {best * 100:.2f}')
                else:
                    print(f'No improvement. Best {self.metric_name}: {best * 100:.2f}')

                self.logger.log_epoch(
                    epoch,
                    train_loss=train_loss,
                    lr=lr_scheduler.get_last_lr()[0],
                    train_time=train_time,
                    metrics=shown,
                    extra={
                        'val_time_s': round(eval_time, 2),
                        f'best_{self.metric_name}': float(best * 100),
                        'is_best': bool(is_best),
                        **self.extra(shown, best * 100),
                    },
                )

            best = self.broadcast(best)

        total_time = time.time() - start_time
        if is_rank0():
            self.save(model, 'last_model.pth')
            self.logger.info(
                f'{self.task} train done | last_model.pth saved | total {total_time:.1f}s')

        if is_dist_env():
            dist.barrier()
            dist.destroy_process_group()

    def train_epoch(self, args, loader, model, optimizer, lr_scheduler, epoch):
        loss_sum, loss_count = 0.0, 0
        start = time.time()

        for step, batch in enumerate(loader):
            loss = self.train_step(args, batch, model)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if is_rank0():
                loss_sum += loss.item()
                loss_count += 1
                if (step + 1) % 10 == 0:
                    print(f'epoch: {epoch} step: [{step + 1}/{len(loader)}] '
                          f'lr: {lr_scheduler.get_last_lr()[0]:.6f} loss: {loss.item():.4f}')

        duration = time.time() - start
        avg_loss = loss_sum / loss_count if loss_count else 0.0
        if is_rank0():
            print(f'\nEpoch {epoch} train loss: {avg_loss:.4f}, time: {duration:.1f}s')
        return avg_loss, duration

    def save(self, model, name):
        state_dict = model.state_dict() if not self.args.distributed \
            else model.module.state_dict()
        torch.save(state_dict, os.path.join(self.args.save_dir, name))

    @staticmethod
    def broadcast(value):
        if not is_dist_env():
            return value
        tensor = torch.tensor(value, dtype=torch.float32, device='cuda')
        dist.broadcast(tensor, src=0)
        return float(tensor.item())
