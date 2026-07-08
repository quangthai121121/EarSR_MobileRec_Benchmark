from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Dict, Any, Union

import numpy as np
import torch
import yaml

ConfigValue = Union[str, int, float, bool, None, Dict[str, Any], List[Any]]


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def experiment_tag(reduce_percent: float) -> str:
    """Tag thí nghiệm: reduce 10% → 'r10', reduce 5% → 'r5'."""
    rp = float(reduce_percent)
    if rp == int(rp):
        return f'r{int(rp)}'
    return f'r{rp:g}'


def reduce_percent_to_keep_ratio(reduce_percent: float) -> float:
    """Giảm X% mỗi chiều → còn (100-X)% → ratio giữ lại."""
    rp = float(reduce_percent)
    if not (0 <= rp < 100):
        raise ValueError(f'reduce_percent phải trong [0, 100), nhận {rp}')
    return (100.0 - rp) / 100.0


def baseline_pipeline_name(reduce_percent: float) -> str:
    return f'lr_{experiment_tag(reduce_percent)}'


def format_with_tag(value: ConfigValue, tag: str) -> ConfigValue:
    if isinstance(value, str):
        if '{tag}' in value:
            return value.format(tag=tag)
        return value
    if isinstance(value, dict):
        return {k: format_with_tag(v, tag) for k, v in value.items()}
    if isinstance(value, list):
        return [format_with_tag(v, tag) for v in value]
    return value


def load_benchmark_config(
    path: str | Path,
    reduce_percent: float | None = None,
) -> Dict[str, Any]:
    """Load YAML và expand {tag} theo reduce_percent (giảm X% kích thước mỗi chiều)."""
    cfg = load_config(path)
    data = cfg.setdefault('data', {})
    rp = reduce_percent if reduce_percent is not None else data.get('reduce_percent')
    if rp is None:
        return cfg
    rp = float(rp)
    data['reduce_percent'] = rp
    tag = experiment_tag(rp)
    data['experiment_tag'] = tag
    data['keep_percent'] = 100.0 - rp
    cfg = format_with_tag(cfg, tag)
    data = cfg['data']
    if not data.get('sr_input_root'):
        data['sr_input_root'] = data['lr_root']
    return cfg


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set seed for Python, NumPy and PyTorch.

    deterministic=True giúp kết quả lặp lại tốt hơn giữa các lần chạy trên
    cùng máy/cùng phiên bản thư viện. Một số CUDA ops vẫn có thể khác rất nhỏ
    giữa GPU/driver khác nhau.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def seed_worker(worker_id: int) -> None:
    """Deterministic DataLoader worker seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_image(path: Path, extensions: Iterable[str]) -> bool:
    return path.suffix.lower() in {e.lower() for e in extensions}


def list_images(root: str | Path, extensions: Iterable[str]) -> List[Path]:
    root = Path(root)
    return [p for p in root.rglob('*') if p.is_file() and is_image(p, extensions)]


def rel_to_root(path: str | Path, root: str | Path) -> Path:
    return Path(path).resolve().relative_to(Path(root).resolve())


def copy_tree_images(src_root: str | Path, dst_root: str | Path, extensions: Iterable[str]) -> int:
    src_root = Path(src_root)
    dst_root = ensure_dir(dst_root)
    count = 0
    for img in list_images(src_root, extensions):
        out = dst_root / rel_to_root(img, src_root)
        ensure_dir(out.parent)
        shutil.copy2(img, out)
        count += 1
    return count


def resolve_device(device_name: str) -> torch.device:
    if device_name == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')
