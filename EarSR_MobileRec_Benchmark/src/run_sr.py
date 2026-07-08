from __future__ import annotations
import argparse
import os
import subprocess
from pathlib import Path

from .common import load_benchmark_config, ensure_dir, set_seed
from .bicubic_sr import bicubic_upscale_dataset


def run_external(method: dict, input_root: Path, output_root: Path, scale: int, seed: int = 42) -> None:
    checkpoint = method.get('checkpoint', '')
    template = method.get('command_template')
    if not template:
        raise ValueError(f"{method['name']} thiếu command_template")
    cmd = template.format(
        scale=scale,
        checkpoint=checkpoint,
        input_dir=str(input_root),
        output_dir=str(output_root),
        repo_dir=method.get('repo_dir', ''),
        seed=seed,
    )
    ensure_dir(output_root)
    env = os.environ.copy()
    env['PYTHONHASHSEED'] = str(seed)
    env.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    print(f'Running external SR: {method["name"]} | seed={seed}')
    print(cmd)
    subprocess.run(cmd, shell=True, check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description='Tạo ảnh SR cho các method trong config.')
    parser.add_argument('--config', default='configs/benchmark.yaml')
    parser.add_argument('--method', default='all', help='all hoặc tên method, ví dụ bicubic_x4')
    parser.add_argument(
        '--reduce_percent',
        type=float,
        default=None,
        help='Giảm X%% kích thước mỗi chiều (override config). VD: 5, 10, 15.',
    )
    args = parser.parse_args()

    cfg = load_benchmark_config(args.config, reduce_percent=args.reduce_percent)
    seed = int(cfg.get('project', {}).get('seed', 42))
    set_seed(seed, deterministic=True)
    data = cfg['data']
    input_root = Path(data.get('sr_input_root') or data.get('lr_root') or data['original_root'])
    sr_root = Path(data['sr_root'])
    scale = int(data.get('scale', 4))
    extensions = data.get('extensions')

    for m in cfg['sr_methods']:
        if not m.get('enabled', False):
            continue
        if args.method != 'all' and m['name'] != args.method:
            continue
        if m['type'] in {'original', 'lr', 'hr_mod'}:
            print(f"Skip {m['name']}: không cần tạo SR cho pipeline type={m['type']}")
            continue
        output_root = sr_root / m['name']
        if m['type'] == 'bicubic':
            n = bicubic_upscale_dataset(input_root, output_root, scale=scale, extensions=extensions)
            print(f'Done bicubic: {n} images -> {output_root}')
        elif m['type'] == 'external':
            run_external(m, input_root, output_root, scale, seed=seed)
        else:
            raise ValueError(f"Unknown SR type: {m['type']}")


if __name__ == '__main__':
    main()
