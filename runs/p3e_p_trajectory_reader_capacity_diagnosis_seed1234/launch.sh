#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/yezhe/伪查询/runs/p3e_p_trajectory_reader_capacity_diagnosis_seed1234
nohup "${ROOT}/run_all.sh" > "${ROOT}/p3e_p_run.log" 2>&1 &
echo $! > "${ROOT}/p3e_p_run.pid"
echo "started pid=$!"

