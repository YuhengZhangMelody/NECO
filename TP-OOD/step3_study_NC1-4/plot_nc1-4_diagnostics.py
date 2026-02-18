#!/usr/bin/env python3
import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from openood.datasets import get_dataloader
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.utils.config import Config, parse_config


@dataclass
class CheckpointItem:
    epoch_label: str
    epoch_num: int
    ckpt_name: str
    ckpt_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Plot NC diagnostics: class-mean distance, within-class variance, and cosine(W, mean).')
    parser.add_argument(
        '--ckpt-root',
        type=str,
        default='results/cifar100_resnet18_32x32_base_e100_lr0.1_default',
        help='Root folder containing seed subfolders like s0/s1/...')
    parser.add_argument(
        '--seed-dirs',
        type=str,
        default='',
        help='Comma-separated seed dirs (e.g., s0,s1,s2). Empty means auto-discover s*.')
    parser.add_argument(
        '--epochs',
        type=str,
        default='10,20,30,40,50,60,70,80,90,100',
        help='Comma-separated epochs to evaluate from model_epoch{epoch}.ckpt')
    parser.add_argument('--include-best', action='store_true')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--eps', type=float, default=1e-12)
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/cifar100_resnet18_32x32_base_e100_lr0.1_default/nc1-4_plots')
    return parser.parse_args()


def build_loader(num_workers: int, batch_size: int):
    cfg_files = [
        ROOT_DIR / 'configs/datasets/cifar100/cifar100.yml',
        ROOT_DIR / 'configs/preprocessors/base_preprocessor.yml',
        ROOT_DIR / 'configs/networks/resnet18_32x32.yml',
        ROOT_DIR / 'configs/pipelines/train/baseline.yml',
    ]
    config = parse_config([Config(str(p)) for p in cfg_files])
    config.num_gpus = 0
    config.num_workers = num_workers
    config.save_output = False
    config.merge_option = 'merge'
    config.dataset.train.batch_size = batch_size
    loader_dict = get_dataloader(config)
    return loader_dict['train']


def parse_seed(seed_dir_name: str) -> Optional[int]:
    m = re.match(r'^s(\d+)$', seed_dir_name)
    if m:
        return int(m.group(1))
    return None


def strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if all(k.startswith('module.') for k in state_dict.keys()):
        return {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict


def load_net(ckpt_path: Path, device: torch.device):
    net = ResNet18_32x32(num_classes=100)
    state = torch.load(str(ckpt_path), map_location='cpu')
    state = strip_module_prefix(state)
    try:
        net.load_state_dict(state, strict=True)
    except RuntimeError:
        net.load_state_dict(state, strict=False)
    net.to(device)
    net.eval()
    return net


def collect_checkpoints(seed_dir: Path, epochs: List[int], include_best: bool):
    items: List[CheckpointItem] = []
    for e in epochs:
        name = f'model_epoch{e}.ckpt'
        items.append(CheckpointItem(str(e), e, name, seed_dir / name))
    if include_best:
        items.append(CheckpointItem('best', 10**9, 'best.ckpt', seed_dir / 'best.ckpt'))
    return items


@torch.no_grad()
def compute_diagnostics(net, loader, device: torch.device, max_samples: int, eps: float):
    num_classes = 100
    class_sum = None
    class_count = torch.zeros(num_classes, dtype=torch.long)

    seen = 0
    for batch in tqdm(loader, desc='Pass1-mean', leave=False):
        data = batch['data'].to(device, non_blocking=True)
        label = batch['label'].long().cpu()
        _, feat = net(data, return_feature=True)
        feat = feat.detach().cpu().double()

        if class_sum is None:
            class_sum = torch.zeros(num_classes, feat.shape[1], dtype=torch.float64)

        for c in torch.unique(label):
            idx = (label == c)
            class_sum[c] += feat[idx].sum(dim=0)
            class_count[c] += idx.sum()

        seen += len(label)
        if max_samples > 0 and seen >= max_samples:
            break

    valid = class_count > 0
    means = torch.zeros_like(class_sum)
    means[valid] = class_sum[valid] / class_count[valid].unsqueeze(1).double()
    means_valid = means[valid]

    # Class mean distances (off-diagonal pairwise L2 distances).
    dmat = torch.cdist(means_valid, means_valid, p=2)
    k = dmat.shape[0]
    if k > 1:
        offdiag_mask = ~torch.eye(k, dtype=torch.bool)
        pair_dists = dmat[offdiag_mask]
        mean_dist = pair_dists.mean().item()
        std_dist = pair_dists.std(unbiased=False).item()
    else:
        mean_dist = float('nan')
        std_dist = float('nan')

    # Within-class variance per class: (1/n_c) sum ||h_i - mu_c||^2
    within_sum = torch.zeros(num_classes, dtype=torch.float64)
    seen = 0
    for batch in tqdm(loader, desc='Pass2-within', leave=False):
        data = batch['data'].to(device, non_blocking=True)
        label = batch['label'].long().cpu()
        _, feat = net(data, return_feature=True)
        feat = feat.detach().cpu().double()

        diff = feat - means[label]
        d2 = diff.pow(2).sum(dim=1)
        within_sum.scatter_add_(0, label, d2)

        seen += len(label)
        if max_samples > 0 and seen >= max_samples:
            break

    within_per_class = within_sum[valid] / class_count[valid].double()
    within_mean = within_per_class.mean().item()
    within_std = within_per_class.std(unbiased=False).item()

    # Cosine similarity between classifier weights and centered class means.
    mu_bar = means_valid.mean(dim=0, keepdim=True)
    centered = means_valid - mu_bar

    fc_weight = net.get_fc_layer().weight.detach().cpu().double()
    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    w_valid = fc_weight[valid_idx]

    w_hat = w_valid / w_valid.norm(dim=1, keepdim=True).clamp_min(eps)
    m_hat = centered / centered.norm(dim=1, keepdim=True).clamp_min(eps)
    cos_per_class = (w_hat * m_hat).sum(dim=1)
    cos_mean = cos_per_class.mean().item()
    cos_std = cos_per_class.std(unbiased=False).item()

    return {
        'num_classes_used': int(valid.sum().item()),
        'num_samples': int(class_count.sum().item()),
        'class_mean_dist_mean': float(mean_dist),
        'class_mean_dist_std': float(std_dist),
        'within_var_mean': float(within_mean),
        'within_var_std': float(within_std),
        'cos_w_mean_mean': float(cos_mean),
        'cos_w_mean_std': float(cos_std),
    }


def make_curve_plot(df_agg: pd.DataFrame, x_col: str, y_col: str, yerr_col: str,
                    title: str, ylabel: str, out_path: Path):
    plt.figure(figsize=(8, 5))
    x = df_agg[x_col].to_numpy()
    y = df_agg[y_col].to_numpy()
    e = df_agg[yerr_col].to_numpy()

    plt.plot(x, y, marker='o')
    plt.fill_between(x, y - e, y + e, alpha=0.2)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    ckpt_root = (ROOT_DIR / args.ckpt_root).resolve() if not os.path.isabs(args.ckpt_root) else Path(args.ckpt_root)
    out_dir = (ROOT_DIR / args.output_dir).resolve() if not os.path.isabs(args.output_dir) else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.seed_dirs.strip():
        seed_dirs = [ckpt_root / s.strip() for s in args.seed_dirs.split(',') if s.strip()]
    else:
        seed_dirs = sorted([p for p in ckpt_root.glob('s*') if p.is_dir()])

    if not seed_dirs:
        raise FileNotFoundError(f'No seed directories found under: {ckpt_root}')

    epochs = [int(x.strip()) for x in args.epochs.split(',') if x.strip()]
    loader = build_loader(args.num_workers, args.batch_size)
    device = torch.device(args.device)

    rows = []
    for seed_dir in seed_dirs:
        seed_name = seed_dir.name
        seed_num = parse_seed(seed_name)
        for item in collect_checkpoints(seed_dir, epochs, args.include_best):
            row = {
                'seed': seed_num if seed_num is not None else seed_name,
                'seed_dir': seed_name,
                'epoch': item.epoch_label,
                'epoch_num': item.epoch_num,
                'checkpoint': item.ckpt_name,
                'checkpoint_path': str(item.ckpt_path),
            }

            if not item.ckpt_path.exists():
                row.update({'status': 'missing'})
                rows.append(row)
                print(f'[missing] {item.ckpt_path}')
                continue

            print(f'[eval] {item.ckpt_path}')
            net = load_net(item.ckpt_path, device)
            metrics = compute_diagnostics(net, loader, device, args.max_samples, args.eps)
            row.update({'status': 'ok', **metrics})
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(['seed_dir', 'epoch_num']).reset_index(drop=True)
    per_seed_csv = out_dir / 'nc1-4_diagnostics_by_seed_epoch.csv'
    df.to_csv(per_seed_csv, index=False)

    df_ok = df[(df['status'] == 'ok') & (df['epoch'].astype(str).str.match(r'^\\d+$'))].copy()
    if len(df_ok) == 0:
        print(f'No valid numeric-epoch rows found. Saved raw table: {per_seed_csv}')
        return

    metric_cols = ['class_mean_dist_mean', 'within_var_mean', 'cos_w_mean_mean']
    agg_rows = []
    for e, sub in df_ok.groupby('epoch_num'):
        agg_rows.append({
            'epoch_num': int(e),
            'class_mean_dist_mean': sub['class_mean_dist_mean'].mean(),
            'class_mean_dist_std_across_seed': sub['class_mean_dist_mean'].std(ddof=0),
            'within_var_mean': sub['within_var_mean'].mean(),
            'within_var_std_across_seed': sub['within_var_mean'].std(ddof=0),
            'cos_w_mean_mean': sub['cos_w_mean_mean'].mean(),
            'cos_w_mean_std_across_seed': sub['cos_w_mean_mean'].std(ddof=0),
            'num_seeds': len(sub),
        })

    df_agg = pd.DataFrame(agg_rows).sort_values('epoch_num').reset_index(drop=True)
    agg_csv = out_dir / 'nc1-4_diagnostics_aggregate.csv'
    df_agg.to_csv(agg_csv, index=False)

    make_curve_plot(
        df_agg,
        x_col='epoch_num',
        y_col='class_mean_dist_mean',
        yerr_col='class_mean_dist_std_across_seed',
        title='Class Mean Distances vs Epoch',
        ylabel='Mean Pairwise Distance',
        out_path=out_dir / 'class_mean_distances_vs_epoch.png')

    make_curve_plot(
        df_agg,
        x_col='epoch_num',
        y_col='within_var_mean',
        yerr_col='within_var_std_across_seed',
        title='Within-class Variance vs Epoch',
        ylabel='Mean Within-class Variance',
        out_path=out_dir / 'within_class_variance_vs_epoch.png')

    make_curve_plot(
        df_agg,
        x_col='epoch_num',
        y_col='cos_w_mean_mean',
        yerr_col='cos_w_mean_std_across_seed',
        title='Cosine Similarity (Classifier Weights vs Means) vs Epoch',
        ylabel='Mean Cosine Similarity',
        out_path=out_dir / 'cosine_w_vs_means_vs_epoch.png')

    print(f'Saved per-seed table: {per_seed_csv}')
    print(f'Saved aggregate table: {agg_csv}')
    print(f'Saved figures under: {out_dir}')


if __name__ == '__main__':
    main()
