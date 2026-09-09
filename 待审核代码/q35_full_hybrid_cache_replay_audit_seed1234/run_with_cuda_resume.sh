#!/usr/bin/env bash
# CUDA-interruption resume wrapper: stage markers in artifacts/stage_markers
# make every stage idempotent, so re-running resumes where it stopped.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-smoke}"
exec /home/yezhe/data/miniconda3/envs/attnkv/bin/python -u "$ROOT/run_pipeline.py" --mode "$MODE"
