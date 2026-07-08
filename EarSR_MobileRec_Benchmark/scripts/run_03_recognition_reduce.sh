#!/usr/bin/env bash
set -e

REDUCE_PERCENT="${1:?Usage: bash scripts/run_03_recognition_reduce.sh REDUCE_PERCENT (vd. 5, 10, 15)}"

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m src.run_benchmark \
  --config configs/benchmark_10p_all_sr_mobile_rec.yaml \
  --reduce_percent "${REDUCE_PERCENT}"
