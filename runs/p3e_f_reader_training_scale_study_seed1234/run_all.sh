#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yezhe/伪查询/runs/p3e_f_reader_training_scale_study_seed1234"
PY="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
MODEL8="/home/yezhe/all_models/models/Qwen/Qwen3-8B"
MODEL4="/home/yezhe/all_models/models/Qwen/Qwen3-4B"
RAW_TRAIN="/home/yezhe/数据集/HotpotQA/raw/hotpot_train_v1.1.json"
P3D3="/home/yezhe/伪查询/runs/p3d3_layer_aligned_native_query_reader_train512_seed1234"
C1="/home/yezhe/伪查询/runs/p3e_c1_learnable_overcomplete_canonical_head_reader_seed1234/formal512/reader/checkpoint_best.pt"
C2="/home/yezhe/伪查询/runs/p3e_c2_learned_head_structured_writer_seed1234/writer_formal512/worker/checkpoint_best.pt"
C2_EVAL="/home/yezhe/伪查询/runs/p3e_c2_learned_head_structured_writer_seed1234/writer_formal512/eval"
OLD_TRAIN_CACHE="/home/yezhe/伪查询/runs/p3e_e_qwen3_5_second_receiver_onboarding_seed1234/cache/train/index.json"
VAL_CACHE="/home/yezhe/伪查询/runs/p3e_d_current_system_performance_check_seed1234/sender_cache/canonical/index.json"

mkdir -p "$ROOT/data" "$ROOT/cache/train" "$ROOT/train1024" "$ROOT/train2048" \
  "$ROOT/eval512" "$ROOT/eval1024" "$ROOT/eval2048" "$ROOT/summary" "$ROOT/audit"

if [[ ! -f "$ROOT/data/SUCCESS.json" ]]; then
  "$PY" "$ROOT/prepare_nested_data.py" \
    --raw-train "$RAW_TRAIN" \
    --existing512 "$P3D3/data/train.jsonl" \
    --validation64 "$P3D3/data/validation.jsonl" \
    --existing-cache-index "$OLD_TRAIN_CACHE" \
    --model "$MODEL8" \
    --out "$ROOT/data" \
    --seed 1234
fi

if [[ ! -f "$ROOT/cache/train/SUCCESS.json" ]]; then
  "$PY" "$ROOT/cache_extended_canonical.py" \
    --model "$MODEL8" \
    --writer "$C2" \
    --data "$ROOT/data/train2048.jsonl" \
    --negatives "$ROOT/data/hard_negatives.json" \
    --existing512-index "$OLD_TRAIN_CACHE" \
    --out "$ROOT/cache/train" \
    --device cuda
fi

"$PY" "$ROOT/import_512_baseline.py" \
  --source-results "$C2_EVAL/per_sample_generation.jsonl" \
  --source-summary "$C2_EVAL/SUCCESS.json" \
  --out "$ROOT/eval512"

if [[ ! -f "$ROOT/train1024/TRAIN_SUCCESS.json" ]]; then
  "$PY" "$ROOT/train_scale_reader.py" \
    --model "$MODEL4" \
    --memory-index "$ROOT/cache/train/index.json" \
    --data "$ROOT/data/train1024.jsonl" \
    --negatives "$ROOT/data/hard_negatives.json" \
    --init-reader "$C1" \
    --out "$ROOT/train1024" \
    --scale 1024 \
    --epochs 20 \
    --seed 1234 \
    --device cuda
fi

if [[ ! -f "$ROOT/train2048/TRAIN_SUCCESS.json" ]]; then
  "$PY" "$ROOT/train_scale_reader.py" \
    --model "$MODEL4" \
    --memory-index "$ROOT/cache/train/index.json" \
    --data "$ROOT/data/train2048.jsonl" \
    --negatives "$ROOT/data/hard_negatives.json" \
    --init-reader "$C1" \
    --out "$ROOT/train2048" \
    --scale 2048 \
    --epochs 20 \
    --seed 1234 \
    --device cuda
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

if [[ ! -f "$ROOT/eval2048/SUCCESS.json" ]]; then
  "$PY" "$ROOT/eval_scale_reader.py" \
    --model "$MODEL4" \
    --memory-index "$VAL_CACHE" \
    --data "$P3D3/data/validation.jsonl" \
    --reader "$ROOT/train2048/checkpoint_best.pt" \
    --out "$ROOT/eval2048" \
    --scale 2048 \
    --seed 1234 \
    --device cuda
fi

"$PY" "$ROOT/build_semantic_review.py" \
  --eval512 "$ROOT/eval512/per_sample_generation.jsonl" \
  --eval1024 "$ROOT/eval1024/per_sample_generation.jsonl" \
  --eval2048 "$ROOT/eval2048/per_sample_generation.jsonl" \
  --out "$ROOT/summary/semantic_review_blinded.csv" \
  --seed 1234

"$PY" "$ROOT/summarize_scale.py" \
  --summary512 "$ROOT/eval512/SUCCESS.json" \
  --summary1024 "$ROOT/eval1024/SUCCESS.json" \
  --summary2048 "$ROOT/eval2048/SUCCESS.json" \
  --semantic-status "$ROOT/summary/semantic_review_blinded.json" \
  --out "$ROOT/summary"

"$PY" "$ROOT/audit.py" --root "$ROOT"
cp "$ROOT/summary/scale_summary.json" "$ROOT/SUCCESS.json"
