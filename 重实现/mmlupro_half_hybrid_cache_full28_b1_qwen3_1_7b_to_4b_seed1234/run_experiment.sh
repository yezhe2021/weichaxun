#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
mkdir -p "$ROOT/logs"
exec "$PYTHON" -u "$ROOT/run_pipeline.py" --config "$ROOT/config.json"
