# Qwen3.5 full-attention KV-only Sender

Independent experiment implementing:

`Qwen3.5-4B 8 full-attention pre-RoPE K/native V -> offset alignment ->
[4,256] reshape [8,128] -> feature translator -> 8-to-36 depth expander ->
target calibration -> frozen R1 Qwen3-4B sparse Reader`.

The Sender never sees the question. The first version intentionally excludes all
DeltaNet pseudo-KV, convolution/recurrent states, and 32-layer hidden residuals.

Variants:

- S0: deterministic reshape plus fixed relative-depth interpolation.
- S1: learned shared low-rank feature/calibration, fixed depth.
- S2: S1 plus learned residual depth mapping.

Stage A uses representation initialization. Stage B is functional-only and gives
all KV consistency losses exactly zero weight. Final gates are diagnostic only.

Run:

```bash
./run_with_cuda_resume.sh smoke
./run_with_cuda_resume.sh development
```
