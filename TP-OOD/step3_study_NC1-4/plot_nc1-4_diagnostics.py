#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Plot NC1-NC4 curves directly from precomputed CSV.')
    parser.add_argument('--csv-path', type=str,
                        default='results/cifar100_resnet18_32x32_base_e100_lr0.1_default/nc1-4_by_ood/nc1-4_farood_all_by_seed_epoch.csv')
    parser.add_argument('--datasets', type=str, default='',
                        help='Optional comma-separated datasets, e.g. cifar100,cifar10')
    parser.add_argument('--seed-dirs', type=str, default='',
                        help='Optional comma-separated seed dirs, e.g. s0,s1,s2')
    parser.add_argument('--epochs', type=str, default='',
                        help='Optional comma-separated epochs, e.g. 10,20,30,...')
    parser.add_argument('--ood-datasets', type=str, default='all',
                        help='OOD datasets to keep for NC5 plot, default all')
    parser.add_argument('--output-dir', type=str,
                        default='TP-OOD/step3_study_NC1-4/outputs/plots')
    return parser.parse_args()


def mean_std_epoch(df: pd.DataFrame, metrics):
    rows = []
    for epoch_num, sub in df.groupby('epoch_num'):
        row = {'epoch_num': int(epoch_num), 'num_seeds': int(sub['seed_dir'].nunique())}
        for m in metrics:
            row[f'{m}_mean'] = sub[m].mean()
            row[f'{m}_std'] = sub[m].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows).sort_values('epoch_num').reset_index(drop=True)


def plot_curve(df_agg: pd.DataFrame, metric: str, out_path: Path, title: str):
    x = df_agg['epoch_num'].to_numpy()
    y = df_agg[f'{metric}_mean'].to_numpy()
    s = df_agg[f'{metric}_std'].to_numpy()

    plt.figure(figsize=(7.5, 4.8))
    plt.plot(x, y, marker='o', linewidth=2)
    plt.fill_between(x, y - s, y + s, alpha=0.2)
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_combined_nc1234(df_agg: pd.DataFrame, out_path: Path, title: str):
    plt.figure(figsize=(9.2, 5.4))
    for metric in ['nc1', 'nc2', 'nc3', 'nc4']:
        if f'{metric}_mean' in df_agg.columns and df_agg[f'{metric}_mean'].notna().any():
            plt.plot(df_agg['epoch_num'], df_agg[f'{metric}_mean'], marker='o', linewidth=2, label=metric.upper())

    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_diagnostics(df_agg: pd.DataFrame, out_path: Path, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].plot(df_agg['epoch_num'], df_agg['class_mean_distance_mean'], marker='o', linewidth=2)
    axes[0].set_title('Class Mean Distances')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Mean Pairwise Distance')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df_agg['epoch_num'], df_agg['within_class_variance_mean'], marker='o', linewidth=2)
    axes[1].set_title('Within-Class Variance')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Trace(Sigma_W)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df_agg['epoch_num'], df_agg['w_mu_cos_mean_mean'], marker='o', linewidth=2)
    axes[2].set_title('Cosine(Weight, Mean)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Mean Cosine Similarity')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()

    csv_path = (ROOT_DIR / args.csv_path).resolve() if not os.path.isabs(args.csv_path) else Path(args.csv_path)
    out_dir = (ROOT_DIR / args.output_dir).resolve() if not os.path.isabs(args.output_dir) else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f'CSV not found: {csv_path}')

    df = pd.read_csv(csv_path)

    required = ['dataset', 'seed_dir', 'epoch', 'epoch_num', 'status', 'nc1', 'nc2', 'nc3', 'nc4']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'CSV missing required columns: {missing}')

    df = df[df['status'] == 'ok'].copy()

    if args.datasets.strip():
        keep_ds = {x.strip() for x in args.datasets.split(',') if x.strip()}
        df = df[df['dataset'].astype(str).isin(keep_ds)].copy()

    if args.seed_dirs.strip():
        keep_seeds = {s.strip() for s in args.seed_dirs.split(',') if s.strip()}
        df = df[df['seed_dir'].astype(str).isin(keep_seeds)].copy()

    df = df[pd.to_numeric(df['epoch'], errors='coerce').notna()].copy()
    df['epoch_num'] = pd.to_numeric(df['epoch_num'], errors='coerce')
    df = df[df['epoch_num'].notna()].copy()
    df['epoch_num'] = df['epoch_num'].astype(int)

    if args.epochs.strip():
        keep_epochs = {int(x.strip()) for x in args.epochs.split(',') if x.strip()}
        df = df[df['epoch_num'].isin(keep_epochs)].copy()

    if len(df) == 0:
        raise ValueError('No rows left after filtering. Check dataset/seed/epoch/status filters.')

    all_filtered = out_dir / 'nc1-4_filtered_rows.csv'
    df.sort_values(['dataset', 'ood_dataset', 'seed_dir', 'epoch_num']).to_csv(all_filtered, index=False)

    metrics_main = ['nc1', 'nc2', 'nc3', 'nc4']

    diag_metrics = []
    for col in ['class_mean_distance', 'within_class_variance', 'w_mu_cos_mean']:
        if col in df.columns:
            diag_metrics.append(col)

    for dataset_name, ds_df in df.groupby('dataset'):
        ds_out = out_dir / str(dataset_name)
        ds_out.mkdir(parents=True, exist_ok=True)

        # NC1-NC4 are OOD-independent; deduplicate repeated rows caused by multiple OOD datasets.
        dedup_cols = ['dataset', 'seed_dir', 'epoch_num']
        dedup_df = ds_df.sort_values(dedup_cols).drop_duplicates(subset=dedup_cols, keep='first').copy()

        agg = mean_std_epoch(dedup_df, metrics_main + diag_metrics)
        agg.to_csv(ds_out / 'nc1-4_aggregate_by_epoch.csv', index=False)

        for metric in metrics_main:
            plot_curve(agg, metric, ds_out / f'{metric}_vs_epoch.png',
                       f'{dataset_name}: {metric.upper()} vs Epoch')

        plot_combined_nc1234(agg, ds_out / 'nc1-4_vs_epoch_combined.png',
                             f'{dataset_name}: NC1-NC4 vs Epoch (Mean Across Seeds)')

        if len(diag_metrics) == 3:
            plot_diagnostics(agg, ds_out / 'nc_diagnostics_triplet.png',
                             f'{dataset_name}: Neural Collapse Diagnostics')

    print(f'Input CSV: {csv_path}')
    print(f'Filtered rows: {all_filtered}')
    print(f'Plots saved under: {out_dir}')


if __name__ == '__main__':
    main()
