#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from openood.datasets import get_dataloader, get_ood_dataloader
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.utils.config import Config, parse_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Plot ID/OOD feature distribution on first two PCA components '
        'with PCA fitted on ID features.')
    parser.add_argument(
        '--ckpt-path',
        type=str,
        required=True,
        help='Path to checkpoint, e.g. results/.../s0/model_epoch100.ckpt')
    parser.add_argument(
        '--id-split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='ID split used to fit PCA.')
    parser.add_argument(
        '--ood-split',
        type=str,
        default='farood',
        choices=['val', 'nearood', 'farood'])
    parser.add_argument(
        '--ood-dataset',
        type=str,
        default='all',
        help='OOD dataset name under split, or "all".')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument(
        '--max-id-samples',
        type=int,
        default=0,
        help='0 means use all ID samples.')
    parser.add_argument(
        '--max-ood-samples',
        type=int,
        default=0,
        help='0 means use all OOD samples.')
    parser.add_argument(
        '--plot-max-id-points',
        type=int,
        default=20000,
        help='Max plotted ID points after PCA (0 means all).')
    parser.add_argument(
        '--plot-max-ood-points',
        type=int,
        default=20000,
        help='Max plotted OOD points after PCA (0 means all).')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed used only for plotting subsampling.')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/cifar100_resnet18_32x32_base_e100_lr0.1_default/pca')
    return parser.parse_args()


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


def build_loaders(num_workers: int, batch_size: int):
    cfg_files = [
        ROOT_DIR / 'configs/datasets/cifar100/cifar100.yml',
        ROOT_DIR / 'configs/datasets/cifar100/cifar100_ood.yml',
        ROOT_DIR / 'configs/preprocessors/base_preprocessor.yml',
        ROOT_DIR / 'configs/networks/resnet18_32x32.yml',
        ROOT_DIR / 'configs/pipelines/test/test_ood.yml',
        ROOT_DIR / 'configs/postprocessors/msp.yml',
    ]
    config = parse_config([Config(str(p)) for p in cfg_files])
    config.num_gpus = 0
    config.num_workers = num_workers
    config.save_output = False
    config.merge_option = 'merge'

    config.dataset.train.batch_size = batch_size
    config.dataset.val.batch_size = batch_size
    config.dataset.test.batch_size = batch_size
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
        raise KeyError(
            f'OOD dataset {dataset} not found in split {split}. available: {available}')
    return {dataset: split_dict[dataset]}


@torch.no_grad()
def collect_id_features(
        net, loader, device: torch.device, max_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    seen = 0
    for batch in tqdm(loader, desc='ID features', leave=False):
        data = batch['data'].to(device, non_blocking=True)
        label = batch['label'].long().cpu().numpy()
        _, feat = net(data, return_feature=True)
        feat = feat.detach().cpu().numpy()
        features.append(feat)
        labels.append(label)
        seen += feat.shape[0]
        if max_samples > 0 and seen >= max_samples:
            break

    x = np.concatenate(features, axis=0)
    y = np.concatenate(labels, axis=0)
    if max_samples > 0 and x.shape[0] > max_samples:
        x = x[:max_samples]
        y = y[:max_samples]
    return x, y


@torch.no_grad()
def collect_ood_features(
        net, loaders: Dict, device: torch.device, max_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    features = []
    source = []
    total = 0
    for ds_name, loader in loaders.items():
        for batch in tqdm(loader, desc=f'OOD features ({ds_name})', leave=False):
            data = batch['data'].to(device, non_blocking=True)
            _, feat = net(data, return_feature=True)
            feat = feat.detach().cpu().numpy()
            features.append(feat)
            source.append(np.array([ds_name] * feat.shape[0], dtype=object))
            total += feat.shape[0]
            if max_samples > 0 and total >= max_samples:
                break
        if max_samples > 0 and total >= max_samples:
            break

    if total == 0:
        raise ValueError('No OOD samples collected.')

    x = np.concatenate(features, axis=0)
    ds = np.concatenate(source, axis=0)
    if max_samples > 0 and x.shape[0] > max_samples:
        x = x[:max_samples]
        ds = ds[:max_samples]
    return x, ds


def maybe_subsample(
        arr1: np.ndarray,
        arr2: np.ndarray,
        max_points: int,
        rng: np.random.Generator):
    if max_points <= 0 or arr1.shape[0] <= max_points:
        return arr1, arr2
    idx = rng.choice(arr1.shape[0], size=max_points, replace=False)
    return arr1[idx], arr2[idx]


def main():
    args = parse_args()
    ckpt_path = (ROOT_DIR / args.ckpt_path).resolve() if not os.path.isabs(args.ckpt_path) else Path(args.ckpt_path)
    out_dir = (ROOT_DIR / args.output_dir).resolve() if not os.path.isabs(args.output_dir) else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    id_loader_dict, ood_loader_dict = build_loaders(args.num_workers, args.batch_size)
    if args.id_split not in id_loader_dict:
        raise KeyError(f'ID split "{args.id_split}" not found. available: {sorted(id_loader_dict.keys())}')
    selected_ood = select_ood_loaders(ood_loader_dict, args.ood_split, args.ood_dataset)

    device = torch.device(args.device)
    net = load_net(ckpt_path, device)

    id_x, id_y = collect_id_features(net, id_loader_dict[args.id_split], device, args.max_id_samples)
    ood_x, ood_src = collect_ood_features(net, selected_ood, device, args.max_ood_samples)

    pca = PCA(n_components=2)
    pca.fit(id_x)
    id_pca = pca.transform(id_x)
    ood_pca = pca.transform(ood_x)

    rng = np.random.default_rng(args.seed)
    id_pca_plot, id_y_plot = maybe_subsample(id_pca, id_y, args.plot_max_id_points, rng)
    ood_pca_plot, ood_src_plot = maybe_subsample(ood_pca, ood_src, args.plot_max_ood_points, rng)

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        id_pca_plot[:, 0], id_pca_plot[:, 1],
        c=id_y_plot, cmap='tab20', s=5, alpha=0.8, linewidths=0, label='ID')
    ax.scatter(
        ood_pca_plot[:, 0], ood_pca_plot[:, 1],
        c='gray', s=5, alpha=0.2, linewidths=0, label='OOD')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(
        f'PCA on ID ({args.id_split}) features | OOD={args.ood_split}/{args.ood_dataset}\n'
        f'ckpt={ckpt_path.name}')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best')

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('ID class label')
    fig.tight_layout()

    safe_ood_name = args.ood_dataset.replace('/', '-')
    fig_path = out_dir / f'pca_id-{args.id_split}_ood-{args.ood_split}-{safe_ood_name}.png'
    fig.savefig(fig_path, dpi=240)
    plt.close(fig)

    id_df = pd.DataFrame({'pc1': id_pca[:, 0], 'pc2': id_pca[:, 1], 'label': id_y})
    ood_df = pd.DataFrame({'pc1': ood_pca[:, 0], 'pc2': ood_pca[:, 1], 'ood_source': ood_src})
    id_csv = out_dir / f'id_pca2_{args.id_split}.csv'
    ood_csv = out_dir / f'ood_pca2_{args.ood_split}_{safe_ood_name}.csv'
    id_df.to_csv(id_csv, index=False)
    ood_df.to_csv(ood_csv, index=False)

    ratio = pca.explained_variance_ratio_
    meta_path = out_dir / f'pca_meta_{args.id_split}_{args.ood_split}_{safe_ood_name}.txt'
    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write(f'checkpoint: {ckpt_path}\n')
        f.write(f'id_split: {args.id_split}\n')
        f.write(f'ood_split: {args.ood_split}\n')
        f.write(f'ood_dataset: {args.ood_dataset}\n')
        f.write(f'id_samples: {id_x.shape[0]}\n')
        f.write(f'ood_samples: {ood_x.shape[0]}\n')
        f.write(f'pc1_explained_variance_ratio: {ratio[0]:.6f}\n')
        f.write(f'pc2_explained_variance_ratio: {ratio[1]:.6f}\n')

    print(f'Figure saved: {fig_path}')
    print(f'ID projection CSV: {id_csv}')
    print(f'OOD projection CSV: {ood_csv}')
    print(f'Meta info: {meta_path}')


if __name__ == '__main__':
    main()
