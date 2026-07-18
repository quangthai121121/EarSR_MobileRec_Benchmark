"""Benchmark adapter: folder-based SAFMN ×4 inference.

Config CLI:
  python external/SAFMN/test.py --scale 4 --model_path ... --input ... --output ...

IMPORTANT — checkpoint matching:
  Config default `SAFMN_NTIRE_ESR_x4.pth` matches the NTIRE2023 ESR submission
  architecture in `NTIRE2023_ESR/models/team15_SAFMN.py` (has `norm.gamma` / GRN),
  NOT `basicsr/archs/safmn_arch.py` (paper / DF2K Efficient SR).

  Use `--arch ntire` (default) for NTIRE weights, `--arch paper` for DF2K/safmn_arch weights.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

REPO_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def _load_module(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load module from {file_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_safmn_class(arch: str):
    if arch == 'ntire':
        path = REPO_DIR / 'NTIRE2023_ESR' / 'models' / 'team15_SAFMN.py'
        if not path.is_file():
            raise FileNotFoundError(f'NTIRE SAFMN arch not found: {path}')
        # team15 imports einops but does not use it; stub if missing.
        if 'einops' not in sys.modules:
            try:
                import einops  # noqa: F401
            except ImportError:
                import types
                stub = types.ModuleType('einops')
                stub.rearrange = lambda x, *a, **k: x  # unused in team15 forward
                sys.modules['einops'] = stub
        mod = _load_module('team15_SAFMN', path)
        return mod.SAFMN

    if arch == 'paper':
        # Avoid basicsr package __init__; only need registry + safmn_arch.
        import types
        basicsr_dir = REPO_DIR / 'basicsr'

        def ensure_pkg(name: str, path: Path) -> None:
            if name not in sys.modules:
                pkg = types.ModuleType(name)
                pkg.__path__ = [str(path)]  # type: ignore[attr-defined]
                sys.modules[name] = pkg

        ensure_pkg('basicsr', basicsr_dir)
        ensure_pkg('basicsr.utils', basicsr_dir / 'utils')
        ensure_pkg('basicsr.archs', basicsr_dir / 'archs')
        _load_module('basicsr.utils.registry', basicsr_dir / 'utils' / 'registry.py')
        mod = _load_module('basicsr.archs.safmn_arch', basicsr_dir / 'archs' / 'safmn_arch.py')
        return mod.SAFMN

    raise ValueError(f'Unknown arch={arch!r}; use ntire|paper')


def detect_arch_from_checkpoint(checkpoint_path: Path) -> str:
    """NTIRE weights contain top-level GRN `norm.gamma`; paper weights use AttBlock LayerNorm."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ('params_ema', 'params', 'state_dict'):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
    if not isinstance(state, dict):
        return 'ntire'
    keys = list(state.keys())
    if any(k == 'norm.gamma' or k.startswith('norm.gamma') for k in keys):
        return 'ntire'
    if any('norm1' in k for k in keys):
        return 'paper'
    # Filename heuristic
    name = checkpoint_path.name.lower()
    if 'ntire' in name:
        return 'ntire'
    return 'paper'


def unwrap_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ('params_ema', 'params', 'state_dict'):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def list_images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob('*')
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


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


def pad_for_safm(lr: torch.Tensor, min_side: int = 64) -> tuple[torch.Tensor, int, int]:
    """Pad LR so SAFM multi-scale 3x3 convs never see maps smaller than 3x3.

    NTIRE team15 SAFM pools to h//16, w//16 at the deepest level (n_levels=4).
    EarVN crops are often tiny; without padding, conv3x3 raises RuntimeError.
    """
    _, _, h, w = lr.shape
    pad_h = max(0, min_side - h)
    pad_w = max(0, min_side - w)
    if pad_h == 0 and pad_w == 0:
        return lr, h, w
    # replicate: safe when pad > spatial size (common on tiny EarVN crops).
    # reflect would fail if pad_h >= h or pad_w >= w.
    lr = torch.nn.functional.pad(lr, (0, pad_w, 0, pad_h), mode='replicate')
    return lr, h, w


def infer_one(model, lr: torch.Tensor, scale: int) -> torch.Tensor:
    lr_pad, h, w = pad_for_safm(lr)
    sr = model(lr_pad)
    return sr[:, :, : h * scale, : w * scale]


def infer_folder(
    model,
    input_root: Path,
    output_root: Path,
    device: torch.device,
    scale: int,
) -> int:
    images = list_images(input_root)
    if not images:
        raise FileNotFoundError(f'No images found under {input_root}')
    output_root.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.inference_mode():
        for img_path in tqdm(images, desc='SAFMN'):
            rel = img_path.relative_to(input_root)
            lr = read_image_tensor(img_path, device)
            sr = infer_one(model, lr, scale=scale)
            save_image_tensor(sr, output_root / rel)
    return len(images)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='SAFMN folder inference adapter')
    parser.add_argument('--scale', type=int, default=4, choices=[2, 3, 4])
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--input', required=True, dest='input_dir')
    parser.add_argument('--output', required=True, dest='output_dir')
    parser.add_argument(
        '--arch',
        default='auto',
        choices=['auto', 'ntire', 'paper'],
        help='ntire=NTIRE2023 team15 (SAFMN_NTIRE_ESR_*.pth); paper=basicsr safmn_arch',
    )
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

    arch = args.arch
    if arch == 'auto':
        arch = detect_arch_from_checkpoint(checkpoint)
    print(f'Using SAFMN arch={arch} | checkpoint={checkpoint}')

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    )

    SAFMN = load_safmn_class(arch)
    model = SAFMN(
        dim=args.dim,
        n_blocks=args.n_blocks,
        ffn_scale=args.ffn_scale,
        upscaling_factor=args.scale,
    )
    state = unwrap_state_dict(torch.load(checkpoint, map_location='cpu'))
    model.load_state_dict(state, strict=True)
    model.to(device)

    count = infer_folder(model, input_root, output_root, device, scale=args.scale)
    print(f'Done. Processed {count} images -> {output_root}')


if __name__ == '__main__':
    main()
