# P2-G1: Qwen3-4B Native Query Reader

This directory implements only step 1 of the reverse KV communication experiment.
It does not train an 8B-to-4B Writer.

## Fixed setup

- Frozen evidence encoder: Qwen3-4B.
- Frozen receiver backbone: Qwen3-4B.
- Trainable module: rank-32 layer-wise Query adapter plus scalar gates.
- Memory: all evidence-token pre-RoPE K and native V from all 36 layers.
- Data: existing 512 train pairs and 64 test pairs with identical base/CF split.
- Training: generation NLL, base/CF answer-swap margin, and correct-vs-shuffled/mismatched ranking.
- Evaluation: full text, correct KV, counterfactual KV, shuffled, mismatched, zero, and Reader-off.
- Hard gate: Native-KV paired counterfactual consistency must be at least 0.90.

## Run

```bash
cd /home/yezhe/伪查询
chmod +x runs/p2g_reverse_qwen3_8b_to_4b_seed1234/run_step1.sh
nohup bash runs/p2g_reverse_qwen3_8b_to_4b_seed1234/run_step1.sh all \
  > runs/p2g_reverse_qwen3_8b_to_4b_seed1234/logs/step1.log 2>&1 &
echo $! > runs/p2g_reverse_qwen3_8b_to_4b_seed1234/step1.pid
```

If CUDA is temporarily unavailable, replace `all` with `wait-cuda`; the launcher
checks every 60 seconds and starts the same pipeline as soon as CUDA returns.

Monitor without reading result metrics:

```bash
bash runs/p2g_reverse_qwen3_8b_to_4b_seed1234/run_step1.sh status
tail -f runs/p2g_reverse_qwen3_8b_to_4b_seed1234/logs/step1.log
```

The final gate file is written under `step1_native_reader/eval/`.
