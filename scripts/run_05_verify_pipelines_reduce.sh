#!/usr/bin/env bash
set -e

REDUCE_PERCENT="${1:?Usage: bash scripts/run_05_verify_pipelines_reduce.sh REDUCE_PERCENT (vd. 5, 10, 15)}"

python -m src.verify_pipelines \
  --config configs/benchmark_10p_all_sr_mobile_rec.yaml \
  --reduce_percent "${REDUCE_PERCENT}"
