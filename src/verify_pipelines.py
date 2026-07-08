from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from .common import load_benchmark_config, list_images, ensure_dir
from .run_benchmark import get_pipeline_roots


def count_split(root: Path, split: str, extensions) -> tuple[int, int]:
    split_root = root / split
    if not split_root.exists():
        return 0, 0
    classes = [p for p in split_root.iterdir() if p.is_dir()]
    imgs = list_images(split_root, extensions)
    return len(classes), len(imgs)


def main():
    parser = argparse.ArgumentParser(description='Kiểm tra các pipeline ảnh có đủ train/val/test và số ảnh hay không.')
    parser.add_argument('--config', default='configs/benchmark_10p_all_sr_mobile_rec.yaml')
    parser.add_argument('--reduce_percent', type=float, default=None)
    args = parser.parse_args()

    cfg = load_benchmark_config(args.config, reduce_percent=args.reduce_percent)
    extensions = cfg['data'].get('extensions', ['.jpg', '.jpeg', '.png', '.bmp', '.webp'])
    roots = get_pipeline_roots(cfg)
    rows = []
    for pipeline, root in roots.items():
        row = {'pipeline': pipeline, 'root': str(root), 'exists': root.exists()}
        for split in ['train', 'val', 'test']:
            n_cls, n_img = count_split(root, split, extensions)
            row[f'{split}_classes'] = n_cls
            row[f'{split}_images'] = n_img
        rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(cfg['data']['results_root']) / 'pipeline_image_counts.csv'
    ensure_dir(out.parent)
    df.to_csv(out, index=False)
    print(df)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
