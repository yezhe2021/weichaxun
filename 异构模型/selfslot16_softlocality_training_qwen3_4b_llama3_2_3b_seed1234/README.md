# 16-slot soft-locality increased-training study

This experiment tests whether a smaller ordered memory with substantially more diverse
training can improve each model's own-slot functional fidelity. It runs Qwen3-4B and
Llama-3.2-3B independently; it does not train a cross-model translator.

The compressor has 16 ordered slots and fixes the locality strength to the best value from
the preceding 64-slot pilot (`lambda=16`):

```text
score(i,t) = q_i K_t / sqrt(128)
             - 16 * abs((t+0.5)/T - (i+0.5)/16)
```

Pooling remains independent per native layer and KV head. K-derived attention weights are
shared for K and V, and neither layers nor heads are mixed. Models are frozen. Training uses
only final-position full-vocabulary KL against each model's native full-context logits;
gold labels are evaluation-only.

Each model uses 1024 train, 128 validation and 128 test examples, effective batch 8,
four true shuffled epochs, 512 optimizer steps, 4096 sample exposures, and LR 1e-3.
Validation is performed every epoch (128 optimizer steps). Best and last checkpoints plus
full training/validation diagnostics are retained.

```bash
cd /home/yezhe/异构模型/selfslot16_softlocality_training_qwen3_4b_llama3_2_3b_seed1234
bash launch.sh
tail -f logs/pipeline.log
cat runs/study/status.json
```
