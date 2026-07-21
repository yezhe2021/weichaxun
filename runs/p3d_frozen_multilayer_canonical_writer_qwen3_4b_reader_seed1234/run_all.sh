#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yezhe/伪查询/runs/p3d_frozen_multilayer_canonical_writer_qwen3_4b_reader_seed1234
P3B=/home/yezhe/伪查询/runs/p3b_multilayer_question_independent_kv_readability_seed1234
P3C=/home/yezhe/伪查询/runs/p3c_question_independent_multilayer_canonical_writer_seed1234
P3A=/home/yezhe/伪查询/runs/p3a_hotpot_canonical_responsibility_seed1234
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python
MODEL4=/home/yezhe/all_models/models/Qwen/Qwen3-4B
MODEL8=/home/yezhe/all_models/models/Qwen/Qwen3-8B
PROTOCOL=${ROOT}/protocol/protocol.json

export PYTHONPATH=${ROOT}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

wait_cuda() {
  until "${PY}" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; do sleep 60; done
}

run_train() {
  local source="$1" mode="$2" out="$3" epochs="$4" init="${5:-}"
  [[ -f "${out}/TRAIN_SUCCESS.json" ]] && return
  wait_cuda
  args=(--model "${MODEL4}" --protocol "${PROTOCOL}" --source "${source}" --out "${out}" --mode "${mode}" --epochs "${epochs}" --seed 1234 --device cuda)
  [[ -n "${init}" ]] && args+=(--init-checkpoint "${init}")
  "${PY}" "${ROOT}/train_p3d_reader.py" "${args[@]}"
}

run_eval() {
  local source="$1" checkpoint="$2" split="$3" out="$4" max_samples="${5:-0}"
  [[ -f "${out}/SUCCESS.json" ]] && return
  wait_cuda
  "${PY}" "${ROOT}/eval_p3d_reader.py" --model "${MODEL4}" --protocol "${PROTOCOL}" \
    --source "${source}" --checkpoint "${checkpoint}" --split "${split}" --out "${out}" \
    --max-samples "${max_samples}" --seed 1234 --device cuda
}

run_all() {
  mkdir -p "${ROOT}/protocol" "${ROOT}/audit" "${ROOT}/gate" "${ROOT}/logs" "${ROOT}/readers" "${ROOT}/baselines"
  if [[ ! -f "${PROTOCOL}" ]]; then
    "${PY}" "${ROOT}/select_p3d_protocol.py" --p3b "${P3B}" --p3c "${P3C}" --out "${PROTOCOL}"
  fi
  if [[ ! -f "${ROOT}/audit/SUCCESS.json" ]]; then
    "${PY}" "${ROOT}/audit_p3d.py" --protocol "${PROTOCOL}" --model4 "${MODEL4}" --out "${ROOT}/audit/SUCCESS.json"
  fi

  run_train canonical16 small "${ROOT}/readers/canonical16/small/train" 60
  run_eval canonical16 "${ROOT}/readers/canonical16/small/train/checkpoint_best.pt" train "${ROOT}/readers/canonical16/small/eval" 16
  if [[ ! -f "${ROOT}/gate/SUCCESS.json" ]]; then
    set +e
    "${PY}" "${ROOT}/check_p3d_gate.py" --eval "${ROOT}/readers/canonical16/small/eval/SUCCESS.json" \
      --train "${ROOT}/readers/canonical16/small/train/TRAIN_SUCCESS.json" --out "${ROOT}/gate/SUCCESS.json"
    gate_status=$?
    set -e
    if [[ ${gate_status} -ne 0 ]]; then echo "P3-D small overfit gate failed; continuing the requested complete diagnostic." >&2; fi
  fi

  run_train canonical16 full "${ROOT}/readers/canonical16/train" 3 "${ROOT}/readers/canonical16/small/train/checkpoint_best.pt"
  run_eval canonical16 "${ROOT}/readers/canonical16/train/checkpoint_best.pt" test "${ROOT}/readers/canonical16/eval"

  run_train native16 full "${ROOT}/readers/native16/train" 3
  run_eval native16 "${ROOT}/readers/native16/train/checkpoint_best.pt" test "${ROOT}/readers/native16/eval"

  run_train canonical36 full "${ROOT}/readers/canonical36/train" 3
  run_eval canonical36 "${ROOT}/readers/canonical36/train/checkpoint_best.pt" test "${ROOT}/readers/canonical36/eval"

  test_cache=$("${PY}" -c "import json; print(json.load(open('${PROTOCOL}', encoding='utf-8'))['canonical16']['canonical_cache']['test'])")
  if [[ ! -f "${ROOT}/baselines/q4_question_only/SUCCESS.json" ]]; then wait_cuda; "${PY}" "${ROOT}/eval_p3d_text_baseline.py" --model-name qwen3_4b --model "${MODEL4}" --cache "${test_cache}" --condition question_only --out "${ROOT}/baselines/q4_question_only" --device cuda; fi
  if [[ ! -f "${ROOT}/baselines/q4_full_text/SUCCESS.json" ]]; then wait_cuda; "${PY}" "${ROOT}/eval_p3d_text_baseline.py" --model-name qwen3_4b --model "${MODEL4}" --cache "${test_cache}" --condition full_text --out "${ROOT}/baselines/q4_full_text" --device cuda; fi
  if [[ ! -f "${ROOT}/baselines/q8_full_text/SUCCESS.json" ]]; then wait_cuda; "${PY}" "${ROOT}/eval_p3d_text_baseline.py" --model-name qwen3_8b --model "${MODEL8}" --cache "${test_cache}" --condition full_text --out "${ROOT}/baselines/q8_full_text" --device cuda; fi
  old_generation=${P3A}/readers/old_synthetic/eval/per_sample_generation.jsonl
  if [[ ! -f "${ROOT}/baselines/old_synthetic_reader/SUCCESS.json" ]]; then
    if [[ -f "${old_generation}" ]]; then
      "${PY}" "${ROOT}/filter_p3a_old_reader.py" --old "${old_generation}" --cache "${test_cache}" --out "${ROOT}/baselines/old_synthetic_reader"
    else
      mkdir -p "${ROOT}/baselines/old_synthetic_reader"
      printf '{"status":"unavailable","reason":"P3-A old synthetic Reader generations were not found"}\n' > "${ROOT}/baselines/old_synthetic_reader/SUCCESS.json"
    fi
  fi
  "${PY}" "${ROOT}/summarize_p3d.py" --root "${ROOT}"
}

status() {
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true
  pgrep -af 'train_p3d_reader.py|eval_p3d_reader.py|eval_p3d_text_baseline.py|run_all.sh all' || true
  find "${ROOT}" -maxdepth 6 -type f \( -name SUCCESS.json -o -name TRAIN_SUCCESS.json \) -print 2>/dev/null | sort
  for log in "${ROOT}"/logs/*.log "${ROOT}"/p3d_run.log; do
    [[ -f "${log}" ]] && { printf '\n== %s ==\n' "$(basename "${log}")"; tail -n 3 "${log}"; }
  done
}

case "${1:-all}" in
  all) run_all ;;
  status) status ;;
  *) echo "Usage: bash run_all.sh {all|status}" >&2; exit 64 ;;
esac
