#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/yezhe/伪查询/runs/p3e_j_canonical_soft_prefix_executability_seed1234"
PY="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
MODEL4="/home/yezhe/all_models/models/Qwen/Qwen3-4B"
P3D3="/home/yezhe/伪查询/runs/p3d3_layer_aligned_native_query_reader_train512_seed1234"
P3EF="/home/yezhe/伪查询/runs/p3e_f_reader_training_scale_study_seed1234"
C2="/home/yezhe/伪查询/runs/p3e_c2_learned_head_structured_writer_seed1234"
P3ED="/home/yezhe/伪查询/runs/p3e_d_current_system_performance_check_seed1234"

TRAIN_DATA="$P3D3/data/train.jsonl"
VAL_DATA="$P3D3/data/validation.jsonl"
TRAIN_CANONICAL="$P3EF/cache/train/index.json"
VAL_CANONICAL="$P3ED/sender_cache/canonical/index.json"
TRAIN_NATIVE="$P3D3/cache/native/train/index.json"
VAL_NATIVE="$P3D3/cache/native/validation/index.json"
WRITER="$C2/writer_formal512/worker/checkpoint_best.pt"
C1="/home/yezhe/伪查询/runs/p3e_c1_learnable_overcomplete_canonical_head_reader_seed1234/formal512/reader/checkpoint_best.pt"

mkdir -p \
  "$ROOT/smoke16/stage_a" \
  "$ROOT/smoke16/stage_b" \
  "$ROOT/formal512/stage_a" \
  "$ROOT/formal512/stage_b" \
  "$ROOT/stage_a_validation64" \
  "$ROOT/eval64" \
  "$ROOT/review" \
  "$ROOT/audit"

for required in \
  "$TRAIN_DATA" "$VAL_DATA" "$TRAIN_CANONICAL" "$VAL_CANONICAL" \
  "$TRAIN_NATIVE" "$VAL_NATIVE" "$WRITER" "$C1"; do
  test -f "$required"
done

if [[ ! -f "$ROOT/smoke16/stage_a/TRAIN_SUCCESS.json" ]]; then
  "$PY" "$ROOT/train_soft_prefix.py" \
    --stage a --mode smoke16 \
    --model "$MODEL4" \
    --canonical-index "$TRAIN_CANONICAL" \
    --native-index "$TRAIN_NATIVE" \
    --writer-checkpoint "$WRITER" \
    --data "$TRAIN_DATA" \
    --out "$ROOT/smoke16/stage_a" \
    --max-samples 16 --epochs 2 --lr 2e-4 \
    --seed 1234 --device cuda
fi

if [[ ! -f "$ROOT/smoke16/stage_b/TRAIN_SUCCESS.json" ]]; then
  "$PY" "$ROOT/train_soft_prefix.py" \
    --stage b --mode smoke16 \
    --model "$MODEL4" \
    --canonical-index "$TRAIN_CANONICAL" \
    --native-index "$TRAIN_NATIVE" \
    --writer-checkpoint "$WRITER" \
    --data "$TRAIN_DATA" \
    --init-checkpoint "$ROOT/smoke16/stage_a/checkpoint_best.pt" \
    --out "$ROOT/smoke16/stage_b" \
    --max-samples 16 --epochs 2 --lr 1e-4 \
    --seed 1234 --device cuda
fi

if [[ ! -f "$ROOT/formal512/stage_a/TRAIN_SUCCESS.json" ]]; then
  "$PY" "$ROOT/train_soft_prefix.py" \
    --stage a --mode formal512 \
    --model "$MODEL4" \
    --canonical-index "$TRAIN_CANONICAL" \
    --native-index "$TRAIN_NATIVE" \
    --writer-checkpoint "$WRITER" \
    --data "$TRAIN_DATA" \
    --out "$ROOT/formal512/stage_a" \
    --max-samples 512 --epochs 5 --lr 2e-4 \
    --seed 1234 --device cuda
fi

if [[ ! -f "$ROOT/stage_a_validation64/SUCCESS.json" ]]; then
  "$PY" "$ROOT/eval_reconstruction.py" \
    --model "$MODEL4" \
    --canonical-index "$VAL_CANONICAL" \
    --native-index "$VAL_NATIVE" \
    --data "$VAL_DATA" \
    --decoder "$ROOT/formal512/stage_a/checkpoint_best.pt" \
    --out "$ROOT/stage_a_validation64" \
    --max-samples 64 --seed 1234 --device cuda
fi

if [[ ! -f "$ROOT/formal512/stage_b/TRAIN_SUCCESS.json" ]]; then
  "$PY" "$ROOT/train_soft_prefix.py" \
    --stage b --mode formal512 \
    --model "$MODEL4" \
    --canonical-index "$TRAIN_CANONICAL" \
    --native-index "$TRAIN_NATIVE" \
    --writer-checkpoint "$WRITER" \
    --data "$TRAIN_DATA" \
    --init-checkpoint "$ROOT/formal512/stage_a/checkpoint_best.pt" \
    --out "$ROOT/formal512/stage_b" \
    --max-samples 512 --epochs 5 --lr 1e-4 \
    --seed 1234 --device cuda
fi

if [[ ! -f "$ROOT/eval64/SUCCESS.json" ]]; then
  "$PY" "$ROOT/eval_soft_prefix.py" \
    --model "$MODEL4" \
    --canonical-index "$VAL_CANONICAL" \
    --native-index "$VAL_NATIVE" \
    --writer-checkpoint "$WRITER" \
    --data "$VAL_DATA" \
    --decoder "$ROOT/formal512/stage_b/checkpoint_best.pt" \
    --c1-reader "$C1" \
    --out "$ROOT/eval64" \
    --max-samples 64 --max-new-tokens 32 \
    --seed 1234 --device cuda
fi

"$PY" "$ROOT/build_semantic_review.py" \
  --results "$ROOT/eval64/per_sample_generation.jsonl" \
  --out "$ROOT/review/semantic_review_blinded.csv" \
  --seed 1234

"$PY" "$ROOT/summarize_diagnosis.py" \
  --evaluation "$ROOT/eval64/SUCCESS.json" \
  --out "$ROOT/diagnosis.json"

"$PY" "$ROOT/audit.py" --root "$ROOT"
cp "$ROOT/eval64/SUCCESS.json" "$ROOT/SUCCESS.json"
