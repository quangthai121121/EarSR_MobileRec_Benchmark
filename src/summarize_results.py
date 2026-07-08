from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from .common import load_benchmark_config, ensure_dir, baseline_pipeline_name


def collect_results(results_root: Path) -> pd.DataFrame:
    rows = []
    for metrics in results_root.glob('*/*/metrics.csv'):
        pipeline = metrics.parent.parent.name
        family = metrics.parent.name
        row = pd.read_csv(metrics).iloc[0].to_dict()
        row['pipeline'] = pipeline
        row['family'] = family
        rows.append(row)
    return pd.DataFrame(rows)


def safe_baseline_name(name: str) -> str:
    return name.replace('/', '_').replace(' ', '_')


def main():
    parser = argparse.ArgumentParser(description='Tổng hợp kết quả recognition và tính delta so với baseline pipeline.')
    parser.add_argument('--config', default='configs/benchmark.yaml')
    parser.add_argument('--baseline', default='auto', help='Pipeline baseline hoặc "auto" (lr_{tag}).')
    parser.add_argument('--reduce_percent', type=float, default=None)
    args = parser.parse_args()

    cfg = load_benchmark_config(args.config, reduce_percent=args.reduce_percent)
    baseline = args.baseline
    if baseline == 'auto':
        baseline = baseline_pipeline_name(cfg['data']['reduce_percent'])
    results_root = ensure_dir(Path(cfg['data']['results_root']))
    df = collect_results(results_root)
    if df.empty:
        print('Chưa có metrics.csv nào.')
        return

    metric_cols = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    missing = [c for c in metric_cols if c not in df.columns]
    if missing:
        raise ValueError(f'Thiếu metric columns trong metrics.csv: {missing}')

    df = df.sort_values(['pipeline', 'family'])
    raw_path = results_root / 'summary_all.csv'
    df.to_csv(raw_path, index=False)

    if baseline not in set(df['pipeline']):
        available = sorted(df['pipeline'].unique())
        raise ValueError(f'Không thấy baseline pipeline={baseline}. Available: {available}')

    base = df[df['pipeline'] == baseline][['family'] + metric_cols].rename(
        columns={c: f'{c}_baseline' for c in metric_cols}
    )
    merged = df.merge(base, on='family', how='left')
    for c in metric_cols:
        merged[f'delta_{c}'] = merged[c] - merged[f'{c}_baseline']

    baseline_tag = safe_baseline_name(baseline)
    delta_path = results_root / f'summary_delta_vs_{baseline_tag}.csv'
    merged.to_csv(delta_path, index=False)

    avg = merged.groupby('pipeline', as_index=False).agg({
        'accuracy': 'mean',
        'precision_macro': 'mean',
        'recall_macro': 'mean',
        'f1_macro': 'mean',
        'delta_accuracy': 'mean',
        'delta_precision_macro': 'mean',
        'delta_recall_macro': 'mean',
        'delta_f1_macro': 'mean',
    }).sort_values('accuracy', ascending=False)
    avg_path = results_root / f'summary_avg_by_pipeline_vs_{baseline_tag}.csv'
    avg.to_csv(avg_path, index=False)

    # Bảng nhỏ đúng mục tiêu: pipeline nào vượt ảnh downsample/baseline.
    comparison_cols = [
        'pipeline', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
        'delta_accuracy', 'delta_precision_macro', 'delta_recall_macro', 'delta_f1_macro'
    ]
    comparison = avg[comparison_cols].copy()
    comparison['better_than_baseline_acc'] = comparison['delta_accuracy'] > 0
    comparison['better_than_baseline_f1'] = comparison['delta_f1_macro'] > 0
    comparison_path = results_root / f'final_comparison_vs_{baseline_tag}.csv'
    comparison.to_csv(comparison_path, index=False)

    print('\nAverage by pipeline:')
    print(comparison)
    print(f'\nSaved:\n- {raw_path}\n- {delta_path}\n- {avg_path}\n- {comparison_path}')


if __name__ == '__main__':
    main()
