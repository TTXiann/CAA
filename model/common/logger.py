import json
import os
import time
from numbers import Number


def flatten_metrics(metrics, prefix=''):
    """Flatten nested metric dicts into ``class_1/F1``-style keys."""
    flat = {}
    for key, value in metrics.items():
        name = f'{prefix}{key}'
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, prefix=f'{name}/'))
        elif isinstance(value, Number) and not isinstance(value, bool):
            flat[name] = float(value)
        else:
            flat[name] = value
    return flat


class ExperimentLogger:
    def __init__(self, exp_dir, log_name='log.txt', metrics_name='metrics.jsonl'):
        os.makedirs(exp_dir, exist_ok=True)
        self.log_path = os.path.join(exp_dir, log_name)
        self.metrics_path = os.path.join(exp_dir, metrics_name)

    def info(self, message, echo=True):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
        if echo:
            print(line)

    def log_epoch(self, epoch, train_loss=None, lr=None, train_time=None,
                  metrics=None, extra=None, echo=False):
        record = {'epoch': int(epoch), 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
        if train_loss is not None:
            record['train_loss'] = round(float(train_loss), 6)
        if lr is not None:
            record['lr'] = float(lr)
        if train_time is not None:
            record['train_time_s'] = round(float(train_time), 2)
        if metrics:
            record['metrics'] = flatten_metrics(metrics)
        if extra:
            record.update(extra)

        with open(self.metrics_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')

        parts = [f'epoch {int(epoch):03d}']
        if train_loss is not None:
            parts.append(f'train_loss {float(train_loss):.4f}')
        if lr is not None:
            parts.append(f'lr {float(lr):.2e}')
        if train_time is not None:
            parts.append(f'train_time {float(train_time):.1f}s')
        if metrics:
            parts.append(' '.join(
                f'{key}={value:.4f}' if isinstance(value, float) else f'{key}={value}'
                for key, value in record['metrics'].items()))
        self.info(' | '.join(parts), echo=echo)
