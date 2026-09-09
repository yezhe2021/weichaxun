# Full-KV Native Reader: Qwen3-1.7B → Qwen3-4B (seed 1234)

This experiment replaces the old sparse/canonical R1 path with a minimal full-context native-KV translation path:

```text
complete question-independent prefix
  → frozen Qwen3-1.7B Sender
  → 28-layer full pre-RoPE K / native V
  → per-target-layer, K/V-separated Writer
  → 36-layer Qwen3-4B-format full pre-RoPE K / native V
  → Qwen3-4B RoPE + differentiable DynamicCache
  → frozen vanilla Qwen3-4B + later Question
  → Answer
```

There is no supporting-token selection, Canonical memory, Reader LoRA, token mixing, or head mixing.

## Writer variants

- `d0`: calibrated nearest source layer followed by 36 independent K and 36 independent V `Linear(128,128,bias=False)` maps.
- `d1`: calibrated fixed two-layer interpolation followed by the same independent feature maps.
- `d2`: calibrated concatenation of five unique neighboring source layers followed by independent `Linear(640,128,bias=False)` maps.

All writers are token independent and head independent. D0/D1 use identity initialization. D2 initializes the block corresponding to D0's nearest layer as identity and all other blocks as zero, so initialized D0 and D2 are exactly equivalent after calibration.

## Required order

Nothing runs automatically. Inspect the code, then invoke stages explicitly:

```bash
python -u run_pipeline.py --config config.json prepare
python -u run_pipeline.py --config config.json audit
python -u run_pipeline.py --config config.json cache --split train --limit 16
python -u run_pipeline.py --config config.json cache --split validation
python -u run_pipeline.py --config config.json cache --split test
python -u run_pipeline.py --config config.json calibrate
python -u run_pipeline.py --config config.json overfit --writer d2
python -u run_pipeline.py --config config.json train --writer d0
python -u run_pipeline.py --config config.json train --writer d1
python -u run_pipeline.py --config config.json train --writer d2
python -u run_pipeline.py --config config.json diagnose --writer d2 --checkpoint checkpoints/quick/d2/stage_a/best.pt
python -u run_pipeline.py --config config.json evaluate --writer d2
```

The protocol audit is the only fail-fast correctness gate. Representation and functional metrics never block later training stages.
Attention-route KL and attention-output cosine are diagnostic-only outputs and never enter the loss or block Stage B.

## Storage policy

- `cache/source17`: full 1.7B prefix KV, needed by Stage A and Stage B.
- `cache/target4`: full native 4B prefix KV, needed by calibration, Stage A, and reconstruction diagnostics.
- Writer predictions are computed online. Only explicitly requested audit samples may be saved.
- Cache files and checkpoints stay on the server and are excluded from Git.
