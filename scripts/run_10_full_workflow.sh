#!/usr/bin/env bash
set -e

REDUCE_PERCENT="${1:?Usage: bash scripts/run_10_full_workflow.sh REDUCE_PERCENT (vd. 5, 10, 15)}"

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "=== Full workflow | giảm ${REDUCE_PERCENT}% mỗi chiều (tag=r${REDUCE_PERCENT}) ==="

bash scripts/run_00_downsample_reduce.sh "${REDUCE_PERCENT}"
bash scripts/run_02_sr_reduce.sh "${REDUCE_PERCENT}"
bash scripts/run_05_verify_pipelines_reduce.sh "${REDUCE_PERCENT}"
bash scripts/run_03_recognition_reduce.sh "${REDUCE_PERCENT}"
bash scripts/run_04_summary_reduce.sh "${REDUCE_PERCENT}"
