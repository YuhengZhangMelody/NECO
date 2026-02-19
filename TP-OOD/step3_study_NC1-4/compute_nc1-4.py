#!/usr/bin/env python3
import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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
        description='Compute NC1-NC4 over checkpoint trajectory on ID train split.')
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
    parser.add_argument(
        '--include-best',
        action='store_true',
        help='Also evaluate best.ckpt when present.')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--max-samples', type=int, default=0,
                        help='Debug option. 0 means use full train set.')
    parser.add_argument('--eps', type=float, default=1e-12)
    parser.add_argument(
        '--output-csv',
        type=str,
        default='TP-OOD/step3_study_NC1-4/outputs/nc1-4_by_seed_epoch.csv')
    return parser.parse_args()

# Conventions in this script:
# - Main NC2 follows neco-mastery/NC_metrics.py (coherence-based).
# - `nc2_etf_fro` is an additional ETF-style diagnostic and not the main NC2.


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
    config.dataset.test.batch_size = batch_size
    loader_dict = get_dataloader(config)
    return loader_dict['test']


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
        path = seed_dir / name
        items.append(
            CheckpointItem(
                epoch_label=str(e),
                epoch_num=e,
                ckpt_name=name,
                ckpt_path=path,
            ))
    if include_best:
        path = seed_dir / 'best.ckpt'
        items.append(
            CheckpointItem(
                epoch_label='best',
                epoch_num=10**9,
                ckpt_name='best.ckpt',
                ckpt_path=path,
            ))
    return items


def parse_seed(seed_dir_name: str) -> Optional[int]:
    m = re.match(r'^s(\d+)$', seed_dir_name)
    if m:
        return int(m.group(1))
    return None


def coherence(v: torch.Tensor) -> float:
    # v is expected to be [d, C] with each column normalized.
    c = v.shape[1]
    if c <= 1:
        return float('nan')
    g = v.T @ v
    g = g + torch.ones((c, c), dtype=v.dtype) / (c - 1)
    g = g - torch.diag(torch.diag(g))
    return (torch.linalg.norm(g, ord=1).item()) / (c * (c - 1))


def mean_pairwise_distance(x: torch.Tensor) -> float:
    # x: [K, d]
    k = x.shape[0]
    if k <= 1:
        return float('nan')
    d = torch.cdist(x, x, p=2)
    tri = torch.triu_indices(k, k, offset=1)
    return d[tri[0], tri[1]].mean().item()


@torch.no_grad()
def compute_nc1234(net, loader, device: torch.device, max_samples: int, eps: float):
    num_classes = 100
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
    k = means_valid.shape[0]
    mu_bar = means_valid.mean(dim=0, keepdim=True)
    centered = means_valid - mu_bar

    # Sigma_B = (1/C) * sum_c \tilde{mu}_c \tilde{mu}_c^T
    sigma_b = (centered.T @ centered) / max(float(k), 1.0)

    # Accumulate Sigma_W numerator: sum_{c,i} (h_{i,c}-mu_c)(h_{i,c}-mu_c)^T
    sw_sum = torch.zeros((means.shape[1], means.shape[1]), dtype=torch.float64)
    seen = 0
    for batch in tqdm(loader, desc='Pass2-within', leave=False):
        data = batch['data'].to(device, non_blocking=True)
        label = batch['label'].long().cpu()
        _, feat = net(data, return_feature=True)
        feat = feat.detach().cpu().double()

        # Use class-wise chunks in the batch to avoid building a huge (B,D,D) tensor.
        for c in torch.unique(label):
            idx = (label == c)
            z = feat[idx] - means[c]
            sw_sum += z.T @ z

        seen += len(label)
        if max_samples > 0 and seen >= max_samples:
            break

    n_total = int(class_count.sum().item())
    sigma_w = sw_sum / max(float(n_total), 1.0)

    # NC1 = (1/C) * Tr(Sigma_W * Sigma_B^\dagger)
    sigma_b_pinv = torch.linalg.pinv(sigma_b, rcond=eps)
    nc1 = (torch.trace(sigma_w @ sigma_b_pinv) / max(float(k), 1.0)).item()

    # M_ in NC_metrics.py is [d, C] (columns are centered class means).
    m_centered = centered.T  # [d, k]
    m_norms = m_centered.norm(dim=0, keepdim=True).clamp_min(eps)
    m_hat_cols = m_centered / m_norms

    # ETF-form NC2 (kept for reference to your previous definition).
    mu_hat = centered / centered.norm(dim=1, keepdim=True).clamp_min(eps)  # [k, d]
    gram = mu_hat @ mu_hat.T
    if k > 1:
        etf = torch.full((k, k), -1.0 / (k - 1), dtype=gram.dtype)
        etf.fill_diagonal_(1.0)
        nc2_etf_fro = torch.linalg.norm(gram - etf, ord='fro').item()
    else:
        nc2_etf_fro = float('nan')

    # NC2 aligned with neco-mastery/NC_metrics.py: coherence(M_/M_norms).
    nc2 = coherence(m_hat_cols)

    # NC3 aligned with neco-mastery/NC_metrics.py:
    # || W^T/||W^T||_F - M_/||M_||_F ||^2
    fc_weight = net.get_fc_layer().weight.detach().cpu().double()  # [C, d]
    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    w_valid = fc_weight[valid_idx]        # [k, d]
    w_t = w_valid.T                       # [d, k]
    normalized_m = m_centered / m_centered.norm(p='fro').clamp_min(eps)
    normalized_w = w_t / w_t.norm(p='fro').clamp_min(eps)
    nc3 = torch.linalg.norm(normalized_w - normalized_m, ord='fro').pow(2).item()

    # Class-wise cosine alignment between classifier weights and centered class means.
    w_row = w_valid
    m_row = centered
    cos = (w_row * m_row).sum(dim=1) / (
        w_row.norm(dim=1).clamp_min(eps) * m_row.norm(dim=1).clamp_min(eps))
    w_mu_cos_mean = cos.mean().item()

    # NC4 aligned with neco-mastery/NC_metrics.py:
    # mismatch rate between NCC prediction and network prediction.
    mismatch_count = 0
    total_count = 0
    class_ids = valid_idx.long()  # maps NCC arg index -> actual class id
    seen = 0
    means_valid = means[valid]  # [k, d], non-centered class means
    for batch in tqdm(loader, desc='Pass3-nc4', leave=False):
        data = batch['data'].to(device, non_blocking=True)
        logits, feat = net(data, return_feature=True)
        feat = feat.detach().cpu().double()           # [B, d]
        pred_nn = logits.argmax(dim=1).detach().cpu()  # [B]

        # NCC prediction: nearest class mean in feature space.
        d2 = torch.cdist(feat, means_valid, p=2)  # [B, k]
        pred_ncc = class_ids[d2.argmin(dim=1)]    # [B]

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
        'class_mean_distance': float(mean_pairwise_distance(means_valid)),
        'w_mu_cos_mean': float(w_mu_cos_mean),
        'nc1': float(nc1),
        'nc2': float(nc2),
        'nc2_etf_fro': float(nc2_etf_fro),
        'nc3': float(nc3),
        'nc4': float(nc4),
    }


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

    loader = build_loader(args.num_workers, args.batch_size)
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
            }

            if not item.ckpt_path.exists():
                row.update({
                    'status': 'missing',
                    'num_classes_used': None,
                    'num_samples': None,
                    'tr_sigma_w': None,
                    'tr_sigma_b': None,
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

            print(f'[eval] {item.ckpt_path}')
            net = load_net(item.ckpt_path, device)
            metrics = compute_nc1234(net, loader, device, args.max_samples, args.eps)
            row.update({'status': 'ok', **metrics})
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(['seed_dir', 'epoch_num']).reset_index(drop=True)
    df.to_csv(out_csv, index=False)

    print('\nSaved NC1-NC4 table:')
    print(out_csv)
    print(df[['seed_dir', 'epoch', 'status', 'class_mean_distance', 'w_mu_cos_mean',
              'nc1', 'nc2', 'nc2_etf_fro', 'nc3', 'nc4']].tail(20).to_string(index=False))


if __name__ == '__main__':
    main()
