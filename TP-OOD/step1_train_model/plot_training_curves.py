#!/usr/bin/env python3
import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Parse OpenOOD training log and plot training curves.')
    parser.add_argument(
        '--log-path',
        type=str,
        default='results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/log.txt',
        help='Path to OpenOOD log.txt')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/plots',
        help='Directory to save parsed CSV and figures')
    return parser.parse_args()


def parse_log(log_path: Path) -> pd.DataFrame:
    # Example line:
    # Epoch 001 | Time     8s | Train Loss 3.4039 | Val Loss 3.538 | Val Acc 14.50
    pattern = re.compile(
        r"Epoch\s+(\d+)\s*\|\s*Time\s+\d+s\s*\|\s*"
        r"Train Loss\s+([0-9]*\.?[0-9]+)\s*\|\s*"
        r"Val Loss\s+([0-9]*\.?[0-9]+)\s*\|\s*"
        r"Val Acc\s+([0-9]*\.?[0-9]+)"
    )

    rows = []
    with log_path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                rows.append({
                    'epoch': int(m.group(1)),
                    'train_loss': float(m.group(2)),
                    'val_loss': float(m.group(3)),
                    'val_acc': float(m.group(4)),
                })

    if not rows:
        raise ValueError(f'No epoch records matched in log: {log_path}')

    df = pd.DataFrame(rows).sort_values('epoch').drop_duplicates('epoch', keep='last')
    return df.reset_index(drop=True)


def plot_curves(df: pd.DataFrame, output_dir: Path) -> None:
    # 1) Loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['train_loss'], marker='', label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], marker='', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training/Validation Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=200)
    plt.close()

    # 2) Val accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(df['epoch'], df['val_acc'], marker='', color='tab:green', label='Val Acc (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'val_acc_curve.png', dpi=200)
    plt.close()

    # 3) Combined (dual-axis)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df['epoch'], df['train_loss'], marker='', label='Train Loss', color='tab:blue')
    ax1.plot(df['epoch'], df['val_loss'], marker='', label='Val Loss', color='tab:orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df['epoch'], df['val_acc'], marker='', label='Val Acc (%)', color='tab:green')
    ax2.set_ylabel('Val Acc (%)')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

    plt.title('Training Curves')
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves_combined.png', dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    log_path = Path(args.log_path)
    if not log_path.is_absolute():
        log_path = (project_root / log_path).resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        raise FileNotFoundError(f'Log file not found: {log_path}')

    df = parse_log(log_path)
    df.to_csv(output_dir / 'training_metrics_by_epoch.csv', index=False)
    plot_curves(df, output_dir)

    print(f'Parsed epochs: {len(df)}')
    print(f'Input log: {log_path}')
    print(f'Outputs saved to: {output_dir}')


if __name__ == '__main__':
    main()
