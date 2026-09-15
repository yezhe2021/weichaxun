#!/usr/bin/env bash
set -euo pipefail

OLD="/home/yezhe/重实现/mmlupro_options_first_fullkv_trajectory_qwen3_1_7b_to_4b_seed1234/cache"
NEW="/home/yezhe/重实现/mmlupro_b1_scale1024_qwen3_1_7b_to_4b_seed1234/cache"

for family in source17 target4; do
  for split in train validation test; do
    mkdir -p "$NEW/$family/$split"
    for source in "$OLD/$family/$split"/*.pt; do
      [[ -e "$source" ]] || continue
      destination="$NEW/$family/$split/$(basename "$source")"
      if [[ ! -e "$destination" ]]; then
        ln "$source" "$destination"
      fi
    done
  done
done

printf 'Anchored cache files linked without tensor duplication.\n'
