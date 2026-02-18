#!/usr/bin/env python3
import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from openood.datasets import get_dataloader, get_ood_dataloader
from openood.networks import get_network
from openood.utils.config import Config, parse_config


@dataclass
class CheckpointItem:
    epoch_label: str
    epoch_num: int
    ckpt_name: str
    ckpt_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compute NC1-NC4 over checkpoint trajectory on ID train split.')
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--ckpt-root', type=str,
                        default='results/cifar100_resnet18_32x32_base_e100_lr0.1_default',
                        help='Root folder containing seed subfolders like s0/s1/...')
    parser.add_argument('--seed-dirs', type=str, default='',
                        help='Comma-separated seed dirs (e.g., s0,s1,s2). Empty means auto-discover s*.')
    parser.add_argument('--epochs', type=str, default='10,20,30,40,50,60,70,80,90,100',
                        help='Comma-separated epochs to evaluate from model_epoch{epoch}.ckpt')
    parser.add_argument('--include-best', action='store_true',
                        help='Also evaluate best.ckpt when present.')

    parser.add_argument('--ood-split', type=str, default='farood',
                        choices=['val', 'nearood', 'farood'])
    parser.add_argument('--ood-dataset', type=str, default='all',
                        help='OOD dataset under split (e.g. cifar10,tin,mnist,svhn,texture,places365), or all.')
    parser.add_argument('--dataset-config', type=str, default='',
                        help='Override dataset yaml path. Default: configs/datasets/{dataset}/{dataset}.yml')
    parser.add_argument('--ood-config', type=str, default='',
                        help='Override OOD yaml path. Default: configs/datasets/{dataset}/{dataset}_ood.yml')
    parser.add_argument('--network-config', type=str, default='',
                        help='Override network yaml path. Default inferred from dataset image size.')

    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--max-samples', type=int, default=0,
                        help='Debug option. 0 means full split.')
    parser.add_argument('--eps', type=float, default=1e-12)
    parser.add_argument('--output-csv', type=str,
                        default='results/cifar100_resnet18_32x32_base_e100_lr0.1_default/nc1-4_by_ood/nc1-4_farood_all_by_seed_epoch.csv')
    return parser.parse_args()


# Conventions in this script:
# - NC1/NC2/NC3/NC4 are aligned with neco-mastery/NC_metrics.py conventions.
# - `nc2_etf_fro` is kept as an extra diagnostic.


def resolve_paths(args: argparse.Namespace):
    ds = args.dataset
    dataset_cfg = Path(args.dataset_config) if args.dataset_config else (ROOT_DIR / f'configs/datasets/{ds}/{ds}.yml')
    ood_cfg = Path(args.ood_config) if args.ood_config else (ROOT_DIR / f'configs/datasets/{ds}/{ds}_ood.yml')

    if args.network_config:
        network_cfg = Path(args.network_config)
    else:
        if ds in ('cifar10', 'cifar100'):
            network_cfg = ROOT_DIR / 'configs/networks/resnet18_32x32.yml'
        else:
            network_cfg = ROOT_DIR / 'configs/networks/resnet18_224x224.yml'

    return dataset_cfg.resolve(), ood_cfg.resolve(), network_cfg.resolve()


def build_id_loader(dataset_cfg: Path, network_cfg: Path, num_workers: int, batch_size: int):
    cfg_files = [
        dataset_cfg,
        ROOT_DIR / 'configs/preprocessors/base_preprocessor.yml',
        network_cfg,
        ROOT_DIR / 'configs/pipelines/train/baseline.yml',
    ]
    config = parse_config([Config(str(p)) for p in cfg_files])
    config.num_gpus = 0
    config.num_workers = num_workers
    config.save_output = False
    config.merge_option = 'merge'
    config.dataset.train.batch_size = batch_size
    loader_dict = get_dataloader(config)
    return loader_dict['train'], config


def build_ood_loaders(dataset_cfg: Path, ood_cfg: Path, network_cfg: Path, num_workers: int, batch_size: int):
    cfg_files = [
        dataset_cfg,
        ood_cfg,
        ROOT_DIR / 'configs/preprocessors/base_preprocessor.yml',
        network_cfg,
        ROOT_DIR / 'configs/pipelines/test/test_ood.yml',
        ROOT_DIR / 'configs/postprocessors/msp.yml',
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
    return id_loader_dict['train'], ood_loader_dict, config


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


def strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if all(k.startswith('module.') for k in state_dict.keys()):
        return {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict


def load_net(ckpt_path: Path, network_config, device: torch.device):
    net = get_network(network_config)
    state = torch.load(str(ckpt_path), map_location='cpu')
    state = strip_module_prefix(state)
    try:
        net.load_state_dict(state, strict=True)
    except RuntimeError:
        net.load_state_dict(state, strict=False)
    net.to(device)
    net.eval()
    return net


def get_fc_weight(net: torch.nn.Module):
    if hasattr(net, 'get_fc_layer'):
        return net.get_fc_layer().weight
    if hasattr(net, 'fc') and hasattr(net.fc, 'weight'):
        return net.fc.weight
    raise AttributeError('Cannot locate FC weight on network. Expected get_fc_layer() or fc.weight.')


def collect_checkpoints(seed_dir: Path, epochs: List[int], include_best: bool):
    items: List[CheckpointItem] = []
    for e in epochs:
        name = f'model_epoch{e}.ckpt'
        path = seed_dir / name
        items.append(CheckpointItem(str(e), e, name, path))
    if include_best:
        path = seed_dir / 'best.ckpt'
        items.append(CheckpointItem('best', 10**9, 'best.ckpt', path))
    return items


def parse_seed(seed_dir_name: str) -> Optional[int]:
    m = re.match(r'^s(\d+)$', seed_dir_name)
    if m:
        return int(m.group(1))
    return None


def coherence(v: torch.Tensor) -> float:
    # v expected shape [d, C], each column normalized.
    c = v.shape[1]
    if c <= 1:
        return float('nan')
    g = v.T @ v
    g = g + torch.ones((c, c), dtype=v.dtype) / (c - 1)
    g = g - torch.diag(torch.diag(g))
    return (torch.linalg.norm(g, ord=1).item()) / (c * (c - 1))


def mean_pairwise_distance(x: torch.Tensor):
    # x: [K, d]
    k = x.shape[0]
    if k <= 1:
        return float('nan')
    d = torch.cdist(x, x, p=2)
    tri = torch.triu_indices(k, k, offset=1)
    return d[tri[0], tri[1]].mean().item()


@torch.no_grad()
def compute_nc1234(net, loader, device: torch.device, num_classes: int, max_samples: int, eps: float):
    class_sum = None
    class_count = torch.zeros(num_classes, dtype=torch.long)

    seen = 0
    for batch in tqdm(loader, desc='Pass1-mean', leave=False):
        data = batch['data'].to(device, non_blocking=True)
        label = batch['label'].long().cpu()
        _, feat = net(data, return_feature=True)
        feat = feat.detach().cpu()
        if class_sum is None:
            class_sum = torch.zeros(num_classes, feat.shape[1], dtype=torch.float32)

        for c in torch.unique(label):
            idx = (label == c)
            class_sum[c] += feat[idx].sum(dim=0)
            class_count[c] += idx.sum()

        seen += len(label)
        if max_samples > 0 and seen >= max_samples:
            break

    valid = class_count > 0
    means = torch.zeros_like(class_sum, dtype=torch.float64)
    means[valid] = class_sum[valid].double() / class_count[valid].unsqueeze(1).double()

    means_valid = means[valid]
    counts_valid = class_count[valid]
    k = means_valid.shape[0]
    mu_bar = means_valid.mean(dim=0, keepdim=True)
    centered = means_valid - mu_bar

    sigma_b = (centered.T @ centered) / max(float(k), 1.0)

    sw_sum = torch.zeros((means.shape[1], means.shape[1]), dtype=torch.float64)
    seen = 0
    for batch in tqdm(loader, desc='Pass2-within', leave=False):
        data = batch['data'].to(device, non_blocking=True)
        label = batch['label'].long().cpu()
        _, feat = net(data, return_feature=True)
        feat = feat.detach().cpu().double()

        for c in torch.unique(label):
            idx = (label == c)
            z = feat[idx] - means[c]
            sw_sum += z.T @ z

        seen += len(label)
        if max_samples > 0 and seen >= max_samples:
            break

    n_total = int(class_count.sum().item())
    sigma_w = sw_sum / max(float(n_total), 1.0)

    sigma_b_pinv = torch.linalg.pinv(sigma_b, rcond=eps)
    nc1 = (torch.trace(sigma_w @ sigma_b_pinv) / max(float(k), 1.0)).item()

    m_centered = centered.T  # [d, k]
    m_norms = m_centered.norm(dim=0, keepdim=True).clamp_min(eps)
    m_hat_cols = m_centered / m_norms

    mu_hat = centered / centered.norm(dim=1, keepdim=True).clamp_min(eps)
    gram = mu_hat @ mu_hat.T
    if k > 1:
        etf = torch.full((k, k), -1.0 / (k - 1), dtype=gram.dtype)
        etf.fill_diagonal_(1.0)
        nc2_etf_fro = torch.linalg.norm(gram - etf, ord='fro').item()
    else:
        nc2_etf_fro = float('nan')

    nc2 = coherence(m_hat_cols)

    fc_weight = get_fc_weight(net).detach().cpu().double()  # [C, d]
    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    w_valid = fc_weight[valid_idx]
    w_t = w_valid.T

    normalized_m = m_centered / m_centered.norm(p='fro').clamp_min(eps)
    normalized_w = w_t / w_t.norm(p='fro').clamp_min(eps)
    nc3 = torch.linalg.norm(normalized_w - normalized_m, ord='fro').pow(2).item()

    # Diagnostic: class-wise cosine alignment between classifier weights and centered class means.
    w_row = w_valid
    m_row = centered
    cos = (w_row * m_row).sum(dim=1) / (
        w_row.norm(dim=1).clamp_min(eps) * m_row.norm(dim=1).clamp_min(eps))
    w_mu_cos_mean = cos.mean().item()

    mismatch_count = 0
    total_count = 0
    class_ids = valid_idx.long()
    seen = 0
    for batch in tqdm(loader, desc='Pass3-nc4', leave=False):
        data = batch['data'].to(device, non_blocking=True)
        logits, feat = net(data, return_feature=True)
        feat = feat.detach().cpu().double()
        pred_nn = logits.argmax(dim=1).detach().cpu()

        d2 = torch.cdist(feat, means_valid, p=2)
        pred_ncc = class_ids[d2.argmin(dim=1)]

        mismatch_count += (pred_ncc != pred_nn).sum().item()
        total_count += pred_nn.numel()

        seen += pred_nn.numel()
        if max_samples > 0 and seen >= max_samples:
            break

    nc4 = mismatch_count / max(float(total_count), 1.0)

    return {
        'num_classes_used': int(k),
        'num_samples': n_total,
        'tr_sigma_w': float(torch.trace(sigma_w).item()),
        'tr_sigma_b': float(torch.trace(sigma_b).item()),
        'within_class_variance': float(torch.trace(sigma_w).item()),
        'class_mean_distance': float(mean_pairwise_distance(means_valid)),
        'w_mu_cos_mean': float(w_mu_cos_mean),
        'nc1': float(nc1),
        'nc2': float(nc2),
        'nc2_etf_fro': float(nc2_etf_fro),
        'nc3': float(nc3),
        'nc4': float(nc4),
    }, means_valid


def main():
    args = parse_args()

    ckpt_root = (ROOT_DIR / args.ckpt_root).resolve() if not os.path.isabs(args.ckpt_root) else Path(args.ckpt_root)
    out_csv = (ROOT_DIR / args.output_csv).resolve() if not os.path.isabs(args.output_csv) else Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    dataset_cfg, ood_cfg, network_cfg = resolve_paths(args)
    if not dataset_cfg.exists():
        raise FileNotFoundError(f'Dataset config not found: {dataset_cfg}')
    if not network_cfg.exists():
        raise FileNotFoundError(f'Network config not found: {network_cfg}')
    if not ood_cfg.exists():
        raise FileNotFoundError(f'OOD config not found: {ood_cfg}')

    if args.seed_dirs.strip():
        seed_dirs = [ckpt_root / s.strip() for s in args.seed_dirs.split(',') if s.strip()]
    else:
        seed_dirs = sorted([p for p in ckpt_root.glob('s*') if p.is_dir()])

    if not seed_dirs:
        raise FileNotFoundError(f'No seed directories found under: {ckpt_root}')

    epochs = [int(x.strip()) for x in args.epochs.split(',') if x.strip()]
    device = torch.device(args.device)

    id_loader, id_cfg = build_id_loader(dataset_cfg, network_cfg, args.num_workers, args.batch_size)
    num_classes = int(id_cfg.dataset.num_classes)

    _, ood_loader_dict, _ = build_ood_loaders(dataset_cfg, ood_cfg, network_cfg,
                                              args.num_workers, args.batch_size)
    ood_loader_map = select_ood_loaders(ood_loader_dict, args.ood_split, args.ood_dataset)

    rows = []
    for seed_dir in seed_dirs:
        seed_name = seed_dir.name
        seed_num = parse_seed(seed_name)
        ckpt_items = collect_checkpoints(seed_dir, epochs, args.include_best)

        for item in ckpt_items:
            base_row = {
                'dataset': args.dataset,
                'seed': seed_num if seed_num is not None else seed_name,
                'seed_dir': seed_name,
                'epoch': item.epoch_label,
                'epoch_num': item.epoch_num,
                'checkpoint': item.ckpt_name,
                'checkpoint_path': str(item.ckpt_path),
                'ood_split': args.ood_split,
            }

            if not item.ckpt_path.exists():
                row = dict(base_row)
                row.update({
                    'ood_dataset': 'none',
                    'status': 'missing',
                    'num_classes_used': None,
                    'num_samples': None,
                    'tr_sigma_w': None,
                    'tr_sigma_b': None,
                    'within_class_variance': None,
                    'class_mean_distance': None,
                    'w_mu_cos_mean': None,
                    'nc1': None,
                    'nc2': None,
                    'nc2_etf_fro': None,
                    'nc3': None,
                    'nc4': None,
                })
                rows.append(row)
                print(f'[missing] {item.ckpt_path}')
                continue

            print(f'[eval] dataset={args.dataset} ckpt={item.ckpt_path}')
            net = load_net(item.ckpt_path, id_cfg.network, device)

            nc1234, _ = compute_nc1234(
                net, id_loader, device, num_classes, args.max_samples, args.eps)

            for ood_name in ood_loader_map.keys():
                row = dict(base_row)
                row['ood_dataset'] = ood_name
                try:
                    row.update(nc1234)
                    row.update({
                        'status': 'ok',
                    })
                except Exception as e:
                    row.update(nc1234)
                    row.update({
                        'status': f'ood_error:{type(e).__name__}',
                        'error': str(e),
                    })
                rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(['dataset', 'seed_dir', 'epoch_num', 'ood_dataset']).reset_index(drop=True)
    df.to_csv(out_csv, index=False)

    print('\nSaved NC1-NC4 table:')
    print(out_csv)
    show_cols = [
        'dataset', 'seed_dir', 'epoch', 'ood_dataset', 'status',
        'nc1', 'nc2', 'nc3', 'nc4'
    ]
    print(df[show_cols].tail(30).to_string(index=False))


if __name__ == '__main__':
    main()
