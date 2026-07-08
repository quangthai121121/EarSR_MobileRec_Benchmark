from __future__ import annotations
from typing import Dict, Any
import timm


def select_available_models(families: Dict[str, Any], pretrained: bool = True) -> Dict[str, str]:
    available = set(timm.list_models(pretrained=pretrained)) | set(timm.list_models(pretrained=False))
    selected = {}
    missing = {}
    for family, info in families.items():
        candidates = info.get('candidates', [])
        chosen = None
        for c in candidates:
            if c in available:
                chosen = c
                break
        if chosen is None:
            # fallback: fuzzy search by family prefix/name
            key = family.lower().replace('-', '').replace('_', '')
            matches = [m for m in sorted(available) if key in m.lower().replace('-', '').replace('_', '')]
            if matches:
                chosen = matches[0]
        if chosen is None:
            missing[family] = candidates
        else:
            selected[family] = chosen
    if missing:
        print('WARNING: Không tìm thấy một số model trong timm hiện tại:')
        for k, v in missing.items():
            print(f'  - {k}: candidates={v}')
        print('Hãy chạy: python -m src.list_timm_models --pattern "*mobile*" --pretrained')
    return selected
