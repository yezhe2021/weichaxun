# Native token0 + hard-chunk16 joint training

This is a one-variable comparison against `token0_plus_slot16_joint_training`: soft locality
is replaced by a strict balanced contiguous partition, while data, slot count, initialization,
training schedule, loss, cache schema and evaluation remain fixed.

For `N=T-1` Native KV tokens after token0:

```text
C_i = [floor(i*N/16), floor((i+1)*N/16))
A_i = softmax(q_i K_Ci^T / sqrt(128))
K_slot_i = A_i K_Ci
V_slot_i = A_i V_Ci
```

Each slot can attend only inside its own chunk, but retains learned content attention within
that chunk. K and V share the same K-derived weights. Layers and KV heads are independent.

The receiver cache is unchanged:

```text
Native token0 -> position 0
hard-chunk slots 0..15 -> positions 1..16
receiver suffix -> positions 17..
```

Qwen3-4B and Llama-3.2-3B are frozen and trained separately. Only slot queries are optimized.
Each model uses 1024/128/128 train/validation/test rows, batch 8, four true epochs, 512 steps,
4096 exposures, LR 1e-3 and final-position full-vocabulary KL. Gold is evaluation-only.

Evaluation reports Accuracy, Native agreement, choice/full KL, the same-position
`without_token0` control, attention diagnostics and actual per-chunk token counts.

```bash
cd /home/yezhe/异构模型/token0_plus_hardchunk16_joint_training_qwen3_4b_llama3_2_3b_seed1234
bash launch.sh
tail -f logs/pipeline.log
cat runs/study/status.json
```
