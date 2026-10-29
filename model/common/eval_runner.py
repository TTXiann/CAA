import json
import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model.common.logger import ExperimentLogger
from model.common.runner import (configure_env, is_dist_env, is_rank0,
                                 print_metrics, scale_metrics, seed_worker,
                                 setup_distributed)


def build_eval_loader(args, dataset):
    g = torch.Generator()
    g.manual_seed(args.seed + (dist.get_rank() if is_dist_env() else 0))

    sampler = None
    if args.distributed:
        sampler = DistributedSampler(
            dataset, shuffle=False, drop_last=False, seed=args.seed)

    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        sampler=sampler, num_workers=args.num_workers, pin_memory=True,
        drop_last=False, worker_init_fn=seed_worker, generator=g,
        persistent_workers=(args.num_workers > 0))


def metrics_lines(metrics):
    lines = []
    for key, value in metrics.items():
        if isinstance(value, dict):
            inner = ' '.join(f'{k}: {v:.4f}' for k, v in value.items())
            lines.append(f'{key} -> {inner}')
        elif isinstance(value, float):
            lines.append(f'{key}: {value:.4f}')
        else:
            lines.append(f'{key}: {value}')
    return lines


class EvalRunner:
    """Shared evaluation pipeline; subclasses only provide task-specific hooks."""

    task = 'task'
    config_keys = ()

    def __init__(self, args):
        self.args = args

    # ----- task hooks -----

    def prepare(self, args):
        """Task-level preparation before the run dir is resolved."""

    def experiment_dir(self, args):
        raise NotImplementedError

    def post_overlay(self, args, cfg):
        """Task-specific handling after the architecture overlay."""

    def build_model(self, args):
        raise NotImplementedError

    def build_dataset(self, args):
        raise NotImplementedError

    def evaluate(self, args, model, loader):
        raise NotImplementedError

    def summary(self, metrics):
        return []

    # ----- shared pipeline -----

    def run(self):
        args = self.args
        configure_env(args.seed)
        setup_distributed(args)
        args.dist_eval = int(bool(args.distributed))

        self.prepare(args)
        args.run_dir = self.experiment_dir(args)
        self.overlay_config(args)

        dataset = self.build_dataset(args)
        loader = build_eval_loader(args, dataset)
        if is_rank0():
            print(f'Split: {args.split}, samples: {len(dataset)}')

        model = self.build_model(args).cuda()
        model.set_finetune()
        model.eval()
        self.load_checkpoint(args, model)

        start = time.time()
        metrics = self.evaluate(args, model, loader)
        elapsed = time.time() - start

        if is_rank0():
            shown = scale_metrics(metrics)
            print_metrics(shown)

            logger = ExperimentLogger(args.run_dir)
            logger.info(
                f'{self.task} eval | checkpoint=best_model.pth split={args.split} '
                f'samples={len(dataset)} batch={args.batch_size} time={elapsed:.2f}s')
            for line in metrics_lines(shown) + self.summary(shown):
                logger.info('  ' + line, echo=False)

        if is_dist_env():
            dist.barrier()
            dist.destroy_process_group()

    def overlay_config(self, args):
        """Architecture values the run name does not encode come from config.json."""
        path = os.path.join(args.run_dir, 'config.json')
        if not os.path.exists(path):
            return
        with open(path) as f:
            cfg = json.load(f)
        for key in self.config_keys:
            if key in cfg:
                setattr(args, key, cfg[key])
        self.post_overlay(args, cfg)

    def load_checkpoint(self, args, model):
        path = os.path.join(args.run_dir, 'best_model.pth')
        if is_rank0():
            print(f'Loading: {path}')

        state = torch.load(path, map_location='cpu')
        state = state.get('model', state) if isinstance(state, dict) else state
        msg = model.load_state_dict(state, strict=False)
        # after the overlay the architecture must match strictly; fail loudly otherwise
        if msg.missing_keys or msg.unexpected_keys:
            raise RuntimeError(
                f'checkpoint does not match the model: missing={msg.missing_keys} '
                f'unexpected={msg.unexpected_keys}')
