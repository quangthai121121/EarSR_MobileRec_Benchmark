#!/usr/bin/env bash
set -e
python -m src.list_timm_models --pattern '*mobilenetv4*' --pretrained || true
python -m src.list_timm_models --pattern '*repvit*' --pretrained || true
python -m src.list_timm_models --pattern '*mobileone*' --pretrained || true
python -m src.list_timm_models --pattern '*fastvit*' --pretrained || true
python -m src.list_timm_models --pattern '*efficientformerv2*' --pretrained || true
python -m src.list_timm_models --pattern '*ghostnetv2*' --pretrained || true
python -m src.list_timm_models --pattern '*mobilevitv2*' --pretrained || true
