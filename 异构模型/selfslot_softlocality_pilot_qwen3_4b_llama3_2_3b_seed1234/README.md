# 64-slot soft-locality pilot

This quick experiment compares the current global content-attention compressor (`lambda=0`)
with ordered soft-local compressors (`lambda=8` and `lambda=16`) for Qwen3-4B and
Llama-3.2-3B independently.

For slot `i` and token `t`:

```text
score(i,t) = q_i K_t / sqrt(128)
             - lambda * abs((t+0.5)/T - (i+0.5)/64)
```

The positional term makes each slot prefer a local normalized context region while retaining
content-dependent selection. Pooling weights are computed from K and shared to pool K and V.
All operations remain per layer and KV head; there is no cross-layer, cross-head or slot mixing.

Every condition uses the same 64 slots, 16 train, 16 validation and 32 test examples,
128 optimizer steps, effective batch 8, 1024 sample exposures and LR 1e-3. Query parameters
start identically across all lambda values. Native KV caches are reused from the completed
training-sufficiency experiment. Models are frozen; gold labels are evaluation-only.

Metrics include full-vocabulary KL, choice-only KL, Native agreement, Accuracy, slot K/V
cosine, attention entropy, effective attended tokens, maximum attention weight, and the mean
normalized distance between each slot center and tokens it reads.

```bash
cd /home/yezhe/异构模型/selfslot_softlocality_pilot_qwen3_4b_llama3_2_3b_seed1234
bash launch.sh
tail -f logs/pipeline.log
cat runs/pilot/status.json
```
