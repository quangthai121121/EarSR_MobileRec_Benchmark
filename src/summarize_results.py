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
    parser = argparse.ArgumentParser(
        description=(
            'Tổng hợp recognition: so sánh từng pipeline với baseline '
            'trên CÙNG recognition family (không lấy trung bình qua các model).'
        )
    )
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

    df = df.sort_values(['pipeline', 'family']).reset_index(drop=True)
    raw_path = results_root / 'summary_all.csv'
    df.to_csv(raw_path, index=False)

    if baseline not in set(df['pipeline']):
        available = sorted(df['pipeline'].unique())
        raise ValueError(f'Không thấy baseline pipeline={baseline}. Available: {available}')

    # Pairwise: cùng recognition family, so pipeline vs baseline.
    base = df[df['pipeline'] == baseline][['family'] + metric_cols].rename(
        columns={c: f'{c}_baseline' for c in metric_cols}
    )
    merged = df.merge(base, on='family', how='left')
    for c in metric_cols:
        merged[f'delta_{c}'] = merged[c] - merged[f'{c}_baseline']

    merged['better_than_baseline_acc'] = merged['delta_accuracy'] > 0
    merged['better_than_baseline_f1'] = merged['delta_f1_macro'] > 0

    baseline_tag = safe_baseline_name(baseline)
    delta_path = results_root / f'summary_delta_vs_{baseline_tag}.csv'
    merged = merged.sort_values(['family', 'pipeline']).reset_index(drop=True)
    merged.to_csv(delta_path, index=False)

    # Bảng claim chính: từng (pipeline, family), không average.
    comparison_cols = [
        'pipeline', 'family',
        'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
        'accuracy_baseline', 'f1_macro_baseline',
        'delta_accuracy', 'delta_precision_macro', 'delta_recall_macro', 'delta_f1_macro',
        'better_than_baseline_acc', 'better_than_baseline_f1',
    ]
    comparison = merged[comparison_cols].copy()
    comparison_path = results_root / f'final_comparison_vs_{baseline_tag}.csv'
    comparison.to_csv(comparison_path, index=False)

    print(f'\nPairwise comparison vs baseline={baseline} (same recognition family):')
    print(comparison.to_string(index=False))
    print(f'\nSaved:\n- {raw_path}\n- {delta_path}\n- {comparison_path}')


if __name__ == '__main__':
    main()
