#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT/logs"
echo "[$(date '+%F %T')] waiting for GPU: free>=25000 MiB and utilization<=20%" >> "$ROOT/logs/launcher.log"

while true; do
  read -r free util < <(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | head -n1 | tr -d ',' )
  if [[ "$free" -ge 25000 && "$util" -le 20 ]]; then
    sleep 10
    read -r free2 util2 < <(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | head -n1 | tr -d ',' )
    if [[ "$free2" -ge 25000 && "$util2" -le 20 ]]; then
      echo "[$(date '+%F %T')] GPU ready: free=${free2}MiB util=${util2}%; launching" >> "$ROOT/logs/launcher.log"
      bash "$ROOT/launch.sh" >> "$ROOT/logs/launcher.log" 2>&1
      exit 0
    fi
  fi
  sleep 30
done
