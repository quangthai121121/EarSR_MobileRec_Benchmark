#!/usr/bin/env bash
set -e
python -m src.summarize_results --config configs/benchmark.yaml --baseline original
python -m src.benchmark_model_speed --config configs/benchmark.yaml --batch_size 1 --iters 100
