#!/usr/bin/env python3
import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from openood.datasets import get_dataloader, get_ood_dataloader
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
        description='Compute NC5 aligned with neco-mastery/NC_metrics.py')
    parser.add_argument('--ckpt-root', type=str,
                        default='results/cifar100_resnet18_32x32_base_e100_lr0.1_default')
    parser.add_argument('--seed-dirs', type=str, default='')
    parser.add_argument('--epochs', type=str, default='10,20,30,40,50,60,70,80,90,100')
    parser.add_argument('--include-best', action='store_true')

    parser.add_argument('--ood-split', type=str, default='farood',
                        choices=['val', 'nearood', 'farood'])
    parser.add_argument('--ood-dataset', type=str, default='texture',
                        help='For nearood/farood: dataset name (e.g., cifar10/tin/mnist/svhn/texture/places365). Use all for full split.')

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--max-id-samples', type=int, default=0)
    parser.add_argument('--max-ood-samples', type=int, default=0)
    parser.add_argument('--eps', type=float, default=1e-12)

    parser.add_argument('--output-csv', type=str,
                        default='results/cifar100_resnet18_32x32_base_e100_lr0.1_default/nc5_by_seed_epoch.csv')
    return parser.parse_args()


def parse_seed(seed_dir_name: str) -> Optional[int]:
    m = re.match(r'^s(\d+)$', seed_dir_name)
    return int(m.group(1)) if m else None


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


def build_loaders(num_workers: int, batch_size: int):
    cfg_files = [
        ROOT_DIR / 'configs/datasets/cifar100/cifar100.yml',
        ROOT_DIR / 'configs/datasets/cifar100/cifar100_ood.yml',
        ROOT_DIR / 'configs/preprocessors/base_preprocessor.yml',
        ROOT_DIR / 'configs/networks/resnet18_32x32.yml',
        ROOT_DIR / 'configs/pipelines/test/test_ood.yml',
    ]
    config = parse_config([Config(str(p)) for p in cfg_files])
    config.num_gpus = 0
    config.num_workers = num_workers
    config.save_output = False
    config.merge_option = 'merge'

    config.dataset.train.batch_size = batch_size
    config.ood_dataset.batch_size = batch_size

    id_loader_dict = get_dataloader(config)
    ood_loader_dict = get_ood_dataloader(config)
    return id_loader_dict, ood_loader_dict


def select_ood_loaders(ood_loader_dict: Dict, split: str, dataset: str):
    if split == 'val':
        return {'val': ood_loader_dict['val']}

    split_dict = ood_loader_dict[split]
    if dataset == 'all':
        return split_dict

    if dataset not in split_dict:
        available = ','.join(sorted(split_dict.keys()))
        raise KeyError(f'OOD dataset {dataset} not found in split {split}. available: {available}')
    return {dataset: split_dict[dataset]}


@torch.no_grad()
def class_means_id_train(net, id_train_loader, device: torch.device, max_samples: int):
    num_classes = 100
    class_sum = None
    class_count = torch.zeros(num_classes, dtype=torch.long)

    seen = 0
    for batch in tqdm(id_train_loader, desc='ID mean', leave=False):
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
    return means[valid], class_count[valid]


@torch.no_grad()
def ood_mean(net, ood_loaders: Dict[str, torch.utils.data.DataLoader], device: torch.device, max_samples: int):
    feat_sum = None
    total = 0

    for name, loader in ood_loaders.items():
        for batch in tqdm(loader, desc=f'OOD mean ({name})', leave=False):
            data = batch['data'].to(device, non_blocking=True)
            _, feat = net(data, return_feature=True)
            feat = feat.detach().cpu().double()

            if feat_sum is None:
                feat_sum = torch.zeros(feat.shape[1], dtype=torch.float64)

            feat_sum += feat.sum(dim=0)
            total += feat.shape[0]

            if max_samples > 0 and total >= max_samples:
                break
        if max_samples > 0 and total >= max_samples:
            break

    if total == 0:
        raise ValueError('No OOD samples collected for NC5 computation.')
    return feat_sum / float(total), total


def compute_nc5(id_means: torch.Tensor, ood_mean_vec: torch.Tensor, eps: float):
    # Align with neco-mastery/NC_metrics.py NC5:
    # mean_c | <mu_c, mu_ood> / (||mu_c|| * ||mu_ood||) |
    ood_norm = torch.linalg.norm(ood_mean_vec).clamp_min(eps)
    id_norms = torch.linalg.norm(id_means, dim=1).clamp_min(eps)
    dots = id_means @ ood_mean_vec
    cos_abs = torch.abs(dots / (id_norms * ood_norm))
    return cos_abs.mean().item(), cos_abs.std(unbiased=False).item(), cos_abs


def main():
    args = parse_args()

    ckpt_root = (ROOT_DIR / args.ckpt_root).resolve() if not os.path.isabs(args.ckpt_root) else Path(args.ckpt_root)
    out_csv = (ROOT_DIR / args.output_csv).resolve() if not os.path.isabs(args.output_csv) else Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.seed_dirs.strip():
        seed_dirs = [ckpt_root / s.strip() for s in args.seed_dirs.split(',') if s.strip()]
    else:
        seed_dirs = sorted([p for p in ckpt_root.glob('s*') if p.is_dir()])

    if not seed_dirs:
        raise FileNotFoundError(f'No seed directories found under: {ckpt_root}')

    epochs = [int(x.strip()) for x in args.epochs.split(',') if x.strip()]

    id_loader_dict, ood_loader_dict = build_loaders(args.num_workers, args.batch_size)
    selected_ood = select_ood_loaders(ood_loader_dict, args.ood_split, args.ood_dataset)
    device = torch.device(args.device)

    rows = []
    for seed_dir in seed_dirs:
        seed_name = seed_dir.name
        seed_num = parse_seed(seed_name)
        ckpt_items = collect_checkpoints(seed_dir, epochs, args.include_best)

        for item in ckpt_items:
            row = {
                'seed': seed_num if seed_num is not None else seed_name,
                'seed_dir': seed_name,
                'epoch': item.epoch_label,
                'epoch_num': item.epoch_num,
                'checkpoint': item.ckpt_name,
                'checkpoint_path': str(item.ckpt_path),
                'ood_split': args.ood_split,
                'ood_dataset': args.ood_dataset,
            }

            if not item.ckpt_path.exists():
                row.update({'status': 'missing', 'nc5': None, 'nc5_std': None,
                            'num_id_classes': None, 'num_id_samples': None, 'num_ood_samples': None})
                rows.append(row)
                print(f'[missing] {item.ckpt_path}')
                continue

            print(f'[eval] {item.ckpt_path}')
            net = load_net(item.ckpt_path, device)

            id_means, id_counts = class_means_id_train(
                net, id_loader_dict['train'], device, args.max_id_samples)
            ood_mu, ood_n = ood_mean(net, selected_ood, device, args.max_ood_samples)

            nc5, nc5_std, _ = compute_nc5(id_means, ood_mu, args.eps)

            row.update({
                'status': 'ok',
                'num_id_classes': int(id_means.shape[0]),
                'num_id_samples': int(id_counts.sum().item()),
                'num_ood_samples': int(ood_n),
                'nc5': float(nc5),
                'nc5_std': float(nc5_std),
            })
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(['seed_dir', 'epoch_num']).reset_index(drop=True)
    df.to_csv(out_csv, index=False)

    print(f'Saved NC5 table: {out_csv}')
    print(df[['seed_dir', 'epoch', 'status', 'ood_split', 'ood_dataset', 'nc5', 'nc5_std']].tail(20).to_string(index=False))


if __name__ == '__main__':
    main()
