#!/usr/bin/env bash
set -e

# Giảm X% kích thước mỗi chiều (vd. 10 → ảnh còn 90% width/height).
# Usage:
#   bash scripts/run_00_downsample_reduce.sh 5
#   bash scripts/run_00_downsample_reduce.sh 10
#   bash scripts/run_00_downsample_reduce.sh 15

REDUCE_PERCENT="${1:?Usage: bash scripts/run_00_downsample_reduce.sh REDUCE_PERCENT (vd. 5, 10, 15)}"

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

TAG="r${REDUCE_PERCENT}"
LR_ROOT="outputs/lr/earvn_${TAG}"
HR_ROOT="outputs/hr_mod/earvn_${TAG}"
REPORT="outputs/reports/downsample_${TAG}_report.csv"

python -m src.downsample_dataset \
  --input_root data/EarVN1.0_split \
  --output_lr_root "${LR_ROOT}" \
  --output_hr_root "${HR_ROOT}" \
  --reduce_percent "${REDUCE_PERCENT}" \
  --interpolation bicubic \
  --output_format png \
  --degradation bicubic \
  --report_csv "${REPORT}" \
  --seed 42

echo "Done. Giảm ${REDUCE_PERCENT}% mỗi chiều → còn $((100 - REDUCE_PERCENT))% kích thước."
echo "LR dataset: ${LR_ROOT}"
