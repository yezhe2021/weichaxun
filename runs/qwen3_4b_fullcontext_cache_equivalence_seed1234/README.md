# Qwen3-4B full-context text/cache equivalence audit

This is a training-free protocol audit. It freezes a balanced 64-example
HotpotQA distractor test subset, renders all ten paragraphs in original order
with the native Qwen3 chat template (`enable_thinking=False`), tokenizes the
complete rendered prompt once, and splits it at the question-text boundary.

Conditions:

- A0 question only
- A1 full-context text
- A2 official `DynamicCache` continuation
- A3-post manually reconstructed cache from captured post-RoPE K/native V
- A3-pre cache reconstructed from k-normed pre-RoPE K after reapplying Qwen3 RoPE
- A4 answer-different, closest-length shuffled native cache

The model is the untouched Qwen3-4B backbone with eager attention in FP32 so
the audit measures protocol equivalence without BF16 reduction noise. There is no
Writer, Reader, LoRA, adapter, optimizer, or parameter update.

Run smoke:

```bash
mkdir -p logs
bash run_with_cuda_resume.sh smoke 2>&1 | tee -a logs/smoke.log
```

Run the fixed 64-example audit:

```bash
bash run_with_cuda_resume.sh development > logs/pipeline.log 2>&1
```

Per-sample records make the audit resumable after a CUDA interruption.
