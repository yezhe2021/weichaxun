#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/yezhe/伪查询/runs/p3e_l_question_conditioned_sender_diagnosis_seed1234
nohup bash "${ROOT}/run_all.sh" > "${ROOT}/p3e_l_run.log" 2>&1 &
echo $! > "${ROOT}/p3e_l_run.pid"
echo "started pid=$(cat "${ROOT}/p3e_l_run.pid")"
