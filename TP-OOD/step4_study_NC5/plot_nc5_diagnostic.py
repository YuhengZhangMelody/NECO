#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Plot NC5 from precomputed CSV.')
    parser.add_argument('--csv-path', type=str,
                        default='results/cifar100_resnet18_32x32_base_e100_lr0.1_default/nc5_by_seed_epoch.csv')
    parser.add_argument('--seed-dirs', type=str, default='')
    parser.add_argument('--epochs', type=str, default='')
    parser.add_argument('--output-dir', type=str,
                        default='results/cifar100_resnet18_32x32_base_e100_lr0.1_default/nc5_plots')
    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = (ROOT_DIR / args.csv_path).resolve() if not os.path.isabs(args.csv_path) else Path(args.csv_path)
    out_dir = (ROOT_DIR / args.output_dir).resolve() if not os.path.isabs(args.output_dir) else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f'CSV not found: {csv_path}')

    df = pd.read_csv(csv_path)
    required_cols = ['seed_dir', 'epoch', 'epoch_num', 'status', 'nc5']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f'CSV missing required columns: {missing_cols}')

    df = df[df['status'] == 'ok'].copy()

    if args.seed_dirs.strip():
        keep_seeds = {s.strip() for s in args.seed_dirs.split(',') if s.strip()}
        df = df[df['seed_dir'].astype(str).isin(keep_seeds)].copy()

    # Keep numeric epochs only by default (exclude best).
    df = df[pd.to_numeric(df['epoch'], errors='coerce').notna()].copy()
    df['epoch_num'] = pd.to_numeric(df['epoch_num'], errors='coerce')
    df = df[df['epoch_num'].notna()].copy()
    df['epoch_num'] = df['epoch_num'].astype(int)

    if args.epochs.strip():
        keep_epochs = {int(x.strip()) for x in args.epochs.split(',') if x.strip()}
        df = df[df['epoch_num'].isin(keep_epochs)].copy()

    if len(df) == 0:
        raise ValueError('No rows left after filtering. Check status/seed/epoch filters.')

    filtered_csv = out_dir / 'nc5_filtered_rows.csv'
    df.sort_values(['seed_dir', 'epoch_num']).to_csv(filtered_csv, index=False)

    agg_rows = []
    for epoch_num, sub in df.groupby('epoch_num'):
        agg_rows.append({
            'epoch_num': int(epoch_num),
            'num_seeds': int(sub['seed_dir'].nunique()),
            'nc5_mean': sub['nc5'].mean(),
            'nc5_std_across_seed': sub['nc5'].std(ddof=0),
        })
    df_agg = pd.DataFrame(agg_rows).sort_values('epoch_num').reset_index(drop=True)

    agg_csv = out_dir / 'nc5_aggregate_by_epoch.csv'
    df_agg.to_csv(agg_csv, index=False)

    plt.figure(figsize=(8, 5))
    x = df_agg['epoch_num'].to_numpy()
    y = df_agg['nc5_mean'].to_numpy()
    s = df_agg['nc5_std_across_seed'].to_numpy()

    plt.plot(x, y, marker='o', linewidth=2, label='NC5 mean')
    plt.fill_between(x, y - s, y + s, alpha=0.2, label='±1 std')
    plt.xlabel('Epoch')
    plt.ylabel('NC5')
    plt.title('NC5 (ID/OOD orthogonality proxy) vs Epoch')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    fig_path = out_dir / 'nc5_vs_epoch.png'
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f'Input CSV: {csv_path}')
    print(f'Filtered rows: {filtered_csv}')
    print(f'Aggregate CSV: {agg_csv}')
    print(f'Figure: {fig_path}')


if __name__ == '__main__':
    main()
