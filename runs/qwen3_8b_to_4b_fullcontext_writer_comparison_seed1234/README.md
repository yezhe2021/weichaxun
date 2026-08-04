# Qwen3-8B to Qwen3-4B full-context Writer comparison

This experiment compares a strong per-layer full-rank Linear Writer against a
structured Full Writer under the frozen full-context protocol. Qwen3-4B is the
Canonical and its Reader is strict Identity. Both model backbones are frozen;
there is no Receiver LoRA and no supporting-fact token selection.

Writers:

- W0 raw Qwen3-8B KV
- W1 shared RMS-only conversion
- W2 72 independent bias-free 1024x1024 Linear maps
- W3 HeadMix + shared-per-layer head MLP + local ±2 LayerMix + calibration

The Full Writer shares one `128→256→128` MLP among the eight heads inside each
layer/K-or-V branch. Explicit 8x8 HeadMix parameters model head identity.

Training:

1. fixed 512/64/64 manifest and complete 36-layer source caches;
2. common RMS and fixed 128 non-Oracle Stage-A positions;
3. 16-sample overfit gates;
4. Stage A representation alignment;
5. F0 generation;
6. direct CE, Stage-A→CE, and Stage-A→CE+KD branches;
7. unified correct/shuffled/no-memory evaluation.

Stage B contains no representation, route, attention, identity, or parameter
tether. Per-sample cache files and stage markers make the pipeline resumable.

```bash
mkdir -p logs
bash run_with_cuda_resume.sh smoke 2>&1 | tee -a logs/smoke.log
bash run_with_cuda_resume.sh development > logs/pipeline.log 2>&1
```
