#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yezhe/伪查询/runs/p3e_i_adapter_augmented_reader_seed1234"
PY="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
MODEL4="/home/yezhe/all_models/models/Qwen/Qwen3-4B"
P3D3="/home/yezhe/伪查询/runs/p3d3_layer_aligned_native_query_reader_train512_seed1234"
P3EF="/home/yezhe/伪查询/runs/p3e_f_reader_training_scale_study_seed1234"
C1="/home/yezhe/伪查询/runs/p3e_c1_learnable_overcomplete_canonical_head_reader_seed1234/formal512/reader/checkpoint_best.pt"
TRAIN_CACHE="$P3EF/cache/train/index.json"
VAL_CACHE="/home/yezhe/伪查询/runs/p3e_d_current_system_performance_check_seed1234/sender_cache/canonical/index.json"

mkdir -p "$ROOT/smoke16" "$ROOT/formal512" "$ROOT/eval64" "$ROOT/review" "$ROOT/audit"

if [[ ! -f "$ROOT/smoke16/TRAIN_SUCCESS.json" ]]; then
  "$PY" "$ROOT/train_lora_reader.py" \
    --model "$MODEL4" \
    --memory-index "$TRAIN_CACHE" \
    --data "$P3D3/data/train.jsonl" \
    --negatives "$P3EF/data/hard_negatives.json" \
    --base-reader "$C1" \
    --out "$ROOT/smoke16" \
    --mode smoke16 \
    --max-samples 16 \
    --epochs 5 \
    --rank 8 \
    --alpha 16 \
    --dropout 0 \
    --seed 1234 \
    --device cuda
fi

if [[ ! -f "$ROOT/formal512/TRAIN_SUCCESS.json" ]]; then
  "$PY" "$ROOT/train_lora_reader.py" \
    --model "$MODEL4" \
    --memory-index "$TRAIN_CACHE" \
    --data "$P3D3/data/train.jsonl" \
    --negatives "$P3EF/data/hard_negatives.json" \
    --base-reader "$C1" \
    --out "$ROOT/formal512" \
    --mode formal512 \
    --max-samples 512 \
    --epochs 5 \
    --rank 8 \
    --alpha 16 \
    --dropout 0 \
    --seed 1234 \
    --device cuda
fi

if [[ ! -f "$ROOT/eval64/SUCCESS.json" ]]; then
  "$PY" "$ROOT/eval_lora_reader.py" \
    --model "$MODEL4" \
    --memory-index "$VAL_CACHE" \
    --data "$P3D3/data/validation.jsonl" \
    --base-reader "$C1" \
    --lora-reader "$ROOT/formal512/checkpoint_best.pt" \
    --out "$ROOT/eval64" \
    --seed 1234 \
    --device cuda
fi

"$PY" "$ROOT/build_semantic_review.py" \
  --results "$ROOT/eval64/per_sample_generation.jsonl" \
  --out "$ROOT/review/semantic_review_blinded.csv" \
  --seed 1234

"$PY" "$ROOT/audit.py" --root "$ROOT"
cp "$ROOT/eval64/SUCCESS.json" "$ROOT/SUCCESS.json"
