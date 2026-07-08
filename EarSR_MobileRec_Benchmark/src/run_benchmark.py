from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from .common import load_benchmark_config, ensure_dir
from .select_timm_models import select_available_models
from .train_recognition import train_one, parse_letterbox_fill, save_recognition_preview


def get_pipeline_roots(cfg):
    data = cfg['data']
    roots = {}
    for m in cfg['sr_methods']:
        if not m.get('enabled', False):
            continue
        if m['type'] == 'original':
            roots[m['name']] = Path(data['original_root'])
        elif m['type'] == 'lr':
            roots[m['name']] = Path(data['lr_root'])
        elif m['type'] == 'hr_mod':
            roots[m['name']] = Path(data['hr_mod_root'])
        else:
            roots[m['name']] = Path(data['sr_root']) / m['name']
    return roots


def main():
    parser = argparse.ArgumentParser(description='Chạy benchmark recognition cho tất cả SR pipelines và mobile backbones.')
    parser.add_argument('--config', default='configs/benchmark.yaml')
    parser.add_argument('--pipelines', nargs='*', default=None, help='Ví dụ: original bicubic_x4 span_x4')
    parser.add_argument('--families', nargs='*', default=None, help='Ví dụ: MobileNetV4 RepViT')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument(
        '--reduce_percent',
        type=float,
        default=None,
        help='Giảm X%% kích thước mỗi chiều (override config data.reduce_percent). VD: 5, 10, 15.',
    )
    args = parser.parse_args()

    cfg = load_benchmark_config(args.config, reduce_percent=args.reduce_percent)
    letterbox_fill = parse_letterbox_fill(cfg['data'].get('letterbox_fill'))
    selected = select_available_models(cfg['recognition_families'], pretrained=cfg['training']['pretrained'])
    if args.families:
        selected = {k: v for k, v in selected.items() if k in args.families}
    print('Recognition models selected:')
    for fam, model_name in selected.items():
        print(f'  {fam}: {model_name}')

    roots = get_pipeline_roots(cfg)
    if args.pipelines:
        roots = {k: v for k, v in roots.items() if k in args.pipelines}

    results_root = ensure_dir(cfg['data']['results_root'])
    save_preview = bool(cfg['data'].get('save_rec_preview', True))
    preview_root_base = Path(cfg['data'].get('rec_preview_root', results_root / 'rec_preview'))
    extensions = cfg['data'].get('extensions', ['.jpg', '.jpeg', '.png', '.bmp', '.webp'])

    all_rows = []
    for pipeline, data_root in roots.items():
        if not data_root.exists():
            print(f'WARNING: bỏ qua {pipeline}, không thấy folder: {data_root}')
            continue

        if save_preview:
            preview_dir = preview_root_base / pipeline
            if preview_dir.exists() and any(preview_dir.rglob('*')):
                print(f'Skip existing recognition preview: {preview_dir}')
            else:
                n = save_recognition_preview(
                    data_root=data_root,
                    output_root=preview_dir,
                    image_size=cfg['data']['image_size'],
                    letterbox_fill=letterbox_fill,
                    extensions=extensions,
                )
                print(f'Saved recognition preview ({n} images): {preview_dir}')

        for family, model_name in selected.items():
            out_dir = results_root / pipeline / family
            metrics_csv = out_dir / 'metrics.csv'
            if metrics_csv.exists():
                print(f'Skip existing result: {metrics_csv}')
                row = pd.read_csv(metrics_csv).iloc[0].to_dict()
            else:
                row = train_one(
                    data_root=data_root,
                    model_name=model_name,
                    output_dir=out_dir,
                    image_size=cfg['data']['image_size'],
                    batch_size=args.batch_size or cfg['training']['batch_size'],
                    epochs=args.epochs or cfg['training']['epochs'],
                    lr=args.lr or cfg['training']['lr'],
                    weight_decay=cfg['training']['weight_decay'],
                    optimizer_name=cfg['training']['optimizer'],
                    patience=cfg['training']['patience'],
                    pretrained=cfg['training']['pretrained'],
                    amp=cfg['training']['amp'],
                    num_workers=cfg['project']['num_workers'],
                    seed=cfg['project']['seed'],
                    device_name=cfg['project']['device'],
                    letterbox_fill=letterbox_fill,
                )
            row['pipeline'] = pipeline
            row['family'] = family
            row['timm_model_name'] = model_name
            all_rows.append(row)

    if all_rows:
        df = pd.DataFrame(all_rows)
        ensure_dir(results_root)
        df.to_csv(results_root / 'raw_summary.csv', index=False)
        print(df[['pipeline', 'family', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro']])


if __name__ == '__main__':
    main()
