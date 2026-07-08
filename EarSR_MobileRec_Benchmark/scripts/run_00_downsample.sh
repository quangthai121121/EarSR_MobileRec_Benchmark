#!/usr/bin/env bash
set -e

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Tạo dataset LR trước khi chạy SR.
# Input yêu cầu dạng ImageFolder:
# data/EarVN1.0_split/train/<subject>/*.jpg
# data/EarVN1.0_split/val/<subject>/*.jpg
# data/EarVN1.0_split/test/<subject>/*.jpg

python -m src.downsample_dataset \
  --input_root data/EarVN1.0_split \
  --output_lr_root outputs/lr/earvn_x4 \
  --output_hr_root outputs/hr_mod/earvn_x4 \
  --scale 4 \
  --interpolation bicubic \
  --output_format png \
  --mod_crop \
  --degradation bicubic \
  --report_csv outputs/reports/downsample_x4_report.csv \
  --seed 42
