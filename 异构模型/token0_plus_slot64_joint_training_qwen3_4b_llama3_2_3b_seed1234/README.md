# Native token0 + 64-slot joint training

This is the capacity-scaled counterpart of the token0 + 16-slot joint experiment. The only
architectural change is increasing the learned soft-local memory from 16 to 64 slots.

For each native layer and KV head:

```text
Native KV token0 ---------------------------> cache position 0
Native KV tokens 1..T-1 -> 64-slot pool ---> cache positions 1..64
Receiver suffix ----------------------------> positions 65..
```

The slot pool excludes token0, avoiding duplicate access to the bypassed anchor. K and V use
the same K-derived attention weights; layers and KV heads remain independent. Qwen3-4B and
Llama-3.2-3B are trained separately and frozen. Only slot queries are optimized.

Training uses final-position full-vocabulary KL against each model's Native full-context
teacher. Gold labels are evaluation-only. Each model uses 1024 train, 128 validation and 128
test samples, batch 8, four true shuffled epochs, 512 optimizer steps and 4096 exposures.

Final evaluation reports both the joint cache and `without_token0`, which removes token0 while
holding the trained slots, their positions and receiver suffix positions fixed.

```bash
cd /home/yezhe/异构模型/token0_plus_slot64_joint_training_qwen3_4b_llama3_2_3b_seed1234
bash launch.sh
tail -f logs/pipeline.log
cat runs/study/status.json
```
