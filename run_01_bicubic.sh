#!/usr/bin/env bash
set -e
python -m src.run_sr --config configs/benchmark.yaml --method bicubic_x4
