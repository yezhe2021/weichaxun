#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yezhe/伪查询/runs/p3e_k_canonical_information_sufficiency_probe_seed1234"
PY="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
MODEL4="/home/yezhe/all_models/models/Qwen/Qwen3-4B"
P3D3="/home/yezhe/伪查询/runs/p3d3_layer_aligned_native_query_reader_train512_seed1234"
P3EF="/home/yezhe/伪查询/runs/p3e_f_reader_training_scale_study_seed1234"
P3ED="/home/yezhe/伪查询/runs/p3e_d_current_system_performance_check_seed1234"

TRAIN_DATA="$P3D3/data/train.jsonl"
VAL_DATA="$P3D3/data/validation.jsonl"
TRAIN_NATIVE="$P3D3/cache/native/train/index.json"
VAL_NATIVE="$P3D3/cache/native/validation/index.json"
TRAIN_CANONICAL="$P3EF/cache/train/index.json"
VAL_CANONICAL="$P3ED/sender_cache/canonical/index.json"

mkdir -p \
  "$ROOT/cache/train" "$ROOT/cache/validation" \
  "$ROOT/smoke16/eval" "$ROOT/formal512" \
  "$ROOT/eval64" "$ROOT/audit"

for required in \
  "$TRAIN_DATA" "$VAL_DATA" "$TRAIN_NATIVE" "$VAL_NATIVE" \
  "$TRAIN_CANONICAL" "$VAL_CANONICAL"; do
  test -f "$required"
done

if [[ ! -f "$ROOT/cache/train/SUCCESS.json" ]]; then
  "$PY" "$ROOT/cache_probe_sidecar.py" \
    --model "$MODEL4" \
    --native-index "$TRAIN_NATIVE" \
    --data "$TRAIN_DATA" \
    --out "$ROOT/cache/train" \
    --max-samples 512 --device cuda
fi

if [[ ! -f "$ROOT/cache/validation/SUCCESS.json" ]]; then
  "$PY" "$ROOT/cache_probe_sidecar.py" \
    --model "$MODEL4" \
    --native-index "$VAL_NATIVE" \
    --data "$VAL_DATA" \
    --out "$ROOT/cache/validation" \
    --max-samples 64 --device cuda
fi

for mode in text native canonical zero; do
  mkdir -p "$ROOT/smoke16/$mode"
  if [[ ! -f "$ROOT/smoke16/$mode/TRAIN_SUCCESS.json" ]]; then
    "$PY" "$ROOT/train_probe.py" \
      --mode "$mode" \
      --canonical-index "$TRAIN_CANONICAL" \
      --native-index "$TRAIN_NATIVE" \
      --sidecar-index "$ROOT/cache/train/index.json" \
      --data "$TRAIN_DATA" \
      --out "$ROOT/smoke16/$mode" \
      --max-samples 16 --epochs 20 \
      --lr 3e-4 --seed 1234 --device cuda
  fi
done

if [[ ! -f "$ROOT/smoke16/eval/SUCCESS.json" ]]; then
  "$PY" "$ROOT/eval_probes.py" \
    --canonical-index "$TRAIN_CANONICAL" \
    --native-index "$TRAIN_NATIVE" \
    --sidecar-index "$ROOT/cache/train/index.json" \
    --data "$TRAIN_DATA" \
    --text-probe "$ROOT/smoke16/text/checkpoint_best.pt" \
    --native-probe "$ROOT/smoke16/native/checkpoint_best.pt" \
    --canonical-probe "$ROOT/smoke16/canonical/checkpoint_best.pt" \
    --zero-probe "$ROOT/smoke16/zero/checkpoint_best.pt" \
    --out "$ROOT/smoke16/eval" \
    --max-samples 16 --seed 1234 --device cuda
fi

for mode in text native canonical zero; do
  mkdir -p "$ROOT/formal512/$mode"
  if [[ ! -f "$ROOT/formal512/$mode/TRAIN_SUCCESS.json" ]]; then
    "$PY" "$ROOT/train_probe.py" \
      --mode "$mode" \
      --canonical-index "$TRAIN_CANONICAL" \
      --native-index "$TRAIN_NATIVE" \
      --sidecar-index "$ROOT/cache/train/index.json" \
      --data "$TRAIN_DATA" \
      --out "$ROOT/formal512/$mode" \
      --max-samples 512 --epochs 5 \
      --lr 2e-4 --seed 1234 --device cuda
  fi
done

if [[ ! -f "$ROOT/eval64/SUCCESS.json" ]]; then
  "$PY" "$ROOT/eval_probes.py" \
    --canonical-index "$VAL_CANONICAL" \
    --native-index "$VAL_NATIVE" \
    --sidecar-index "$ROOT/cache/validation/index.json" \
    --data "$VAL_DATA" \
    --text-probe "$ROOT/formal512/text/checkpoint_best.pt" \
    --native-probe "$ROOT/formal512/native/checkpoint_best.pt" \
    --canonical-probe "$ROOT/formal512/canonical/checkpoint_best.pt" \
    --zero-probe "$ROOT/formal512/zero/checkpoint_best.pt" \
    --out "$ROOT/eval64" \
    --max-samples 64 --seed 1234 --device cuda
fi

"$PY" "$ROOT/summarize_diagnosis.py" \
  --evaluation "$ROOT/eval64/SUCCESS.json" \
  --out "$ROOT/diagnosis.json"

"$PY" "$ROOT/audit.py" --root "$ROOT"
cp "$ROOT/eval64/SUCCESS.json" "$ROOT/SUCCESS.json"
