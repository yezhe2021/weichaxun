# P2-A2: Query and Output Adapter Ablation

P2-A2 keeps the Qwen3-8B sender and receiver frozen and reads complete,
uncompressed, same-model pre-RoPE K and native V at all 36 receiver layers.

It compares three configurations under identical data, losses, and evaluation:

| Configuration | Query adapter | Output adapter | Scalar gate |
|---|---:|---:|---:|
| `output_only` | 0 | rank 32 | trained |
| `query_only` | rank 32 | 0 | trained |
| `query_output` | rank 32 | rank 32 | trained |

The Query adapter operates on the flattened native pre-RoPE query and learns
where to read. The Output adapter learns how to transform the evidence readout,
and the gate learns the injection magnitude.

## Controlled training

- 512 balanced strict counterfactual training pairs by default.
- Independent balanced schedules for target people, organizations, base cities,
  and counterfactual cities.
- Base/counterfactual generation and answer-swap margin losses.
- Correct-memory versus shuffled/mismatched-memory answer-probability ranking.
- Pair-sharded lazy KV loading to avoid retaining the full cache in RAM.

## Unified evaluation

Each configuration reports free-running generations for correct,
counterfactual, shuffled, mismatched, zero, Reader-off, and aligned
`full_text_prefilled_final` base/counterfactual conditions. Negative-condition
EM is explicitly labeled original-answer leakage rather than accuracy.

Run:

```bash
cd /home/yezhe/伪查询
PY=/home/yezhe/data/miniconda3/envs/attnkv/bin/python \
  bash runs/p2a2_query_output_native_kv_qwen3_8b_seed1234/run_all.sh all
```

Main comparison:

```text
runs/p2a2_query_output_native_kv_qwen3_8b_seed1234/comparison/SUCCESS.json
runs/p2a2_query_output_native_kv_qwen3_8b_seed1234/comparison/comparison.csv
```
