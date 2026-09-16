#!/usr/bin/env bash
set -euo pipefail

FORWARD_ROOT="/home/yezhe/异构模型/openbookqa_official_full16_mlp_train4096_l1_to_q4_seed1234"
REVERSE_ROOT="/home/yezhe/异构模型/openbookqa_official_full36_mlp_train4096_q4_to_l1_seed1234"
PYTHON_BIN="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"

echo "[$(date '+%F %T')] Waiting for forward Llama1B-to-Qwen4B pipeline"
while true; do
    if grep -q '"status": "completed"' "$FORWARD_ROOT/runs/study/status.json" 2>/dev/null; then
        echo "[$(date '+%F %T')] Forward completed; starting reverse pipeline"
        break
    fi
    if grep -q '"status": "failed"' "$FORWARD_ROOT/runs/study/status.json" 2>/dev/null; then
        echo "[$(date '+%F %T')] Forward failed; reverse pipeline will not start"
        exit 2
    fi
    if [[ -f "$FORWARD_ROOT/runs/study/pipeline.pid" ]]; then
        FORWARD_PID=$(<"$FORWARD_ROOT/runs/study/pipeline.pid")
        if ! kill -0 "$FORWARD_PID" 2>/dev/null; then
            echo "[$(date '+%F %T')] Forward process disappeared without a terminal status"
            exit 3
        fi
    fi
    sleep 60
done

cd "$REVERSE_ROOT"
exec "$PYTHON_BIN" -u run_pipeline.py --mode study all
