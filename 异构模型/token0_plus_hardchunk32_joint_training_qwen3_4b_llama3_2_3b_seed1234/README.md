# Native token0 + hard-chunk32 joint training

This is the slot-count-scaled counterpart of `token0_plus_hardchunk16_joint_training`.
The only intended change is increasing the strict contiguous memory from 16 to 32 slots.

For `N=T-1` Native KV tokens after token0:

```text
C_i = [floor(i*N/32), floor((i+1)*N/32))
A_i = softmax(q_i K_Ci^T / sqrt(128))
K_slot_i = A_i K_Ci
V_slot_i = A_i V_Ci
```

Each slot can attend only inside its own chunk but retains learned content attention there.
K and V share K-derived weights; layers and KV heads remain independent.

The receiver schema is:

```text
Native token0 -> position 0
hard-chunk slots 0..31 -> positions 1..32
receiver suffix -> positions 33..
```

About 2% of current rows contain fewer than 32 post-token0 tokens. Their mathematically empty
chunks emit zero K/V and the corresponding cache positions are masked in receiver attention.
This preserves the exact 1024/128/128 manifests instead of filtering or replacing samples.
Empty-chunk rates and actual chunk sizes are recorded.

Qwen3-4B and Llama-3.2-3B are frozen and trained separately. Only slot queries are optimized.
All other settings match hard-chunk16: batch 8, four nominal shuffled passes, 512 optimizer
steps, LR 1e-3, 4096 exposures, final-position full-vocabulary KL and gold only for eval.

```bash
cd /home/yezhe/异构模型/token0_plus_hardchunk32_joint_training_qwen3_4b_llama3_2_3b_seed1234
bash launch.sh
tail -f logs/pipeline.log
cat runs/study/status.json
```
