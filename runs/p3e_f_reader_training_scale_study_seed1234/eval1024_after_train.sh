#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yezhe/伪查询/runs/p3e_f_reader_training_scale_study_seed1234"
PY="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
MODEL4="/home/yezhe/all_models/models/Qwen/Qwen3-4B"
P3D3="/home/yezhe/伪查询/runs/p3d3_layer_aligned_native_query_reader_train512_seed1234"
VAL_CACHE="/home/yezhe/伪查询/runs/p3e_d_current_system_performance_check_seed1234/sender_cache/canonical/index.json"

while pgrep -f "train_scale_reader.py.*--scale 1024" >/dev/null; do
  sleep 60
done

if [[ ! -f "$ROOT/train1024/TRAIN_SUCCESS.json" ]]; then
  echo "train1024 stopped without TRAIN_SUCCESS.json; evaluation was not started"
  exit 1
fi

if [[ ! -f "$ROOT/eval1024/SUCCESS.json" ]]; then
  "$PY" "$ROOT/eval_scale_reader.py" \
    --model "$MODEL4" \
    --memory-index "$VAL_CACHE" \
    --data "$P3D3/data/validation.jsonl" \
    --reader "$ROOT/train1024/checkpoint_best.pt" \
    --out "$ROOT/eval1024" \
    --scale 1024 \
    --seed 1234 \
    --device cuda
fi

echo "eval1024 complete"
