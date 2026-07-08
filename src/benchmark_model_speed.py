from __future__ import annotations
import argparse
import time
from pathlib import Path
import pandas as pd
import torch
import timm

from .common import load_config, resolve_device, ensure_dir
from .select_timm_models import select_available_models


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def measure(model, device, image_size=224, batch_size=1, warmup=20, iters=100):
    model.eval()
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    for _ in range(warmup):
        _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    ms_per_batch = elapsed / iters * 1000
    fps = batch_size * 1000 / ms_per_batch
    return ms_per_batch, fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/benchmark.yaml')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--iters', type=int, default=100)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg['project']['device'])
    selected = select_available_models(cfg['recognition_families'], pretrained=cfg['training']['pretrained'])
    rows = []
    for family, model_name in selected.items():
        try:
            model = timm.create_model(model_name, pretrained=False, num_classes=1000).to(device)
            params = count_params(model)
            ms, fps = measure(model, device, cfg['data']['image_size'], args.batch_size, iters=args.iters)
            rows.append({'family': family, 'model_name': model_name, 'params': params, 'params_M': params / 1e6, 'ms_per_batch': ms, 'fps': fps})
            print(rows[-1])
        except Exception as e:
            print(f'ERROR {family}/{model_name}: {e}')
    out = ensure_dir(cfg['data']['results_root']) / 'recognition_speed.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
