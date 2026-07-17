"""Benchmark adapter: folder-based SAFMN x{2,3,4} inference.

Matches config CLI:
  python external/SAFMN/test.py --scale 4 --model_path ... --input ... --output ...

Uses official SAFMN(dim=36, n_blocks=8, ffn_scale=2.0) from basicsr/archs/safmn_arch.py
(as in inference/inference_safmn.py and NTIRE2023_ESR team15).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

REPO_DIR = Path(__file__).resolve().parent
BASICSR_DIR = REPO_DIR / 'basicsr'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def _ensure_pkg(name: str, path: Path) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is None:
        mod = types.ModuleType(name)
        mod.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = mod
    return mod


def _load_module(fullname: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(fullname, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load module from {file_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    spec.loader.exec_module(module)
    return module


def load_safmn_class():
    """Load SAFMN arch without executing full basicsr/__init__.py."""
    _ensure_pkg('basicsr', BASICSR_DIR)
    _ensure_pkg('basicsr.utils', BASICSR_DIR / 'utils')
    _ensure_pkg('basicsr.archs', BASICSR_DIR / 'archs')
    _load_module('basicsr.utils.registry', BASICSR_DIR / 'utils' / 'registry.py')
    safmn_mod = _load_module('basicsr.archs.safmn_arch', BASICSR_DIR / 'archs' / 'safmn_arch.py')
    return safmn_mod.SAFMN


def list_images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob('*')
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_state_dict(model: torch.nn.Module, checkpoint_path: Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(ckpt, dict):
        for key in ('params_ema', 'params', 'state_dict'):
            if key in ckpt:
                ckpt = ckpt[key]
                break
    model.load_state_dict(ckpt, strict=True)


def read_image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f'Failed to read image: {path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)


def save_image_tensor(tensor: torch.Tensor, path: Path) -> None:
    arr = tensor.detach().float().cpu().clamp(0, 1).squeeze(0).numpy()
    arr = (arr.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), bgr):
        raise IOError(f'Failed to write image: {path}')


def infer_folder(
    model: torch.nn.Module,
    input_root: Path,
    output_root: Path,
    device: torch.device,
) -> int:
    images = list_images(input_root)
    if not images:
        raise FileNotFoundError(f'No images found under {input_root}')

    output_root.mkdir(parents=True, exist_ok=True)
    model.eval()

    with torch.inference_mode():
        for img_path in tqdm(images, desc='SAFMN'):
            rel = img_path.relative_to(input_root)
            out_path = output_root / rel
            lr = read_image_tensor(img_path, device)
            sr = model(lr)
            save_image_tensor(sr, out_path)

    return len(images)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='SAFMN folder inference adapter')
    parser.add_argument('--scale', type=int, default=4, choices=[2, 3, 4])
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--input', required=True, dest='input_dir')
    parser.add_argument('--output', required=True, dest='output_dir')
    parser.add_argument('--dim', type=int, default=36)
    parser.add_argument('--n_blocks', type=int, default=8)
    parser.add_argument('--ffn_scale', type=float, default=2.0)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    checkpoint = Path(args.model_path)

    if not input_root.is_dir():
        raise FileNotFoundError(f'Input folder not found: {input_root}')
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    )

    SAFMN = load_safmn_class()
    model = SAFMN(
        dim=args.dim,
        n_blocks=args.n_blocks,
        ffn_scale=args.ffn_scale,
        upscaling_factor=args.scale,
    )
    load_state_dict(model, checkpoint)
    model.to(device)

    count = infer_folder(model, input_root, output_root, device)
    print(f'Done. Processed {count} images -> {output_root}')


if __name__ == '__main__':
    main()
