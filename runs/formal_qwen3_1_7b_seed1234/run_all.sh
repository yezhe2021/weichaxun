#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/yezhe/伪查询"
PYTHON="/home/yezhe/data/miniconda3/envs/attnkv/bin/python"
MODEL="/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B"
ROOT="$PROJECT/runs/formal_qwen3_1_7b_seed1234"

cd "$PROJECT"
mkdir -p "$ROOT" "$ROOT/controls" "$ROOT/train" "$ROOT/eval" "$ROOT/summary"

run_controls() {
  "$PYTHON" translated_kv_diagnostics.py \
    --model "$MODEL" \
    --max-samples 64 --max-context-tokens 256 --max-new-tokens 16 \
    --methods native,noise,token_shuffle,head_shuffle,low_rank,rope_shift \
    --residual-alphas 0.25,0.5,0.75,1.0 \
    --rank 16 --noise-sigma 0.1 --rope-shift 4 \
    --seed 1234 --device cuda --dtype float16 \
    --out "$ROOT/controls"
}

train_one() {
  local name="$1"
  shift
  "$PYTHON" train_kv_translator.py \
    --model "$MODEL" \
    --max-train-samples 512 --max-val-samples 64 \
    --max-context-tokens 256 --bottleneck 128 --hidden 512 \
    --epochs 1 --seed 1234 --device cuda --dtype float16 \
    --out "$ROOT/train/$name" "$@"
}

eval_one() {
  local name="$1"
  "$PYTHON" translated_kv_diagnostics.py \
    --model "$MODEL" \
    --max-samples 64 --max-context-tokens 256 --max-new-tokens 16 \
    --methods native,translator \
    --translator-checkpoint "$ROOT/train/$name/checkpoint_epoch1.pt" \
    --residual-alphas 0.25,0.5,0.75,1.0 \
    --seed 1234 --device cuda --dtype float16 \
    --out "$ROOT/eval/$name"
}

case "${1:-all}" in
  controls)
    run_controls
    ;;
  train)
    train_one autoencoder --translator-kind autoencoder --objective mse
    train_one mse_only --translator-kind pseudo_sender --objective mse
    train_one ce_only --translator-kind pseudo_sender --objective ce
    train_one mse_ce --translator-kind pseudo_sender --objective mse_ce
    train_one rope_mse_ce --translator-kind pseudo_sender --objective mse_ce --rope-disentangled
    ;;
  eval)
    eval_one autoencoder
    eval_one mse_only
    eval_one ce_only
    eval_one mse_ce
    eval_one rope_mse_ce
    ;;
  all)
    run_controls
    "$0" train
    "$0" eval
    ;;
  *)
    echo "usage: $0 [controls|train|eval|all]" >&2
    exit 2
    ;;
esac
