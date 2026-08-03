# Qwen3-8B full-context text/cache equivalence audit

This is the Qwen3-8B counterpart of the frozen Qwen3-4B protocol audit. It
uses the exact same balanced 64-example HotpotQA distractor sample IDs, prompt,
chat-template settings, one-pass tokenization boundary, and A0-A4 conditions.

Qwen3-8B is loaded without any adapter in FP16 with eager attention. FP16 is
used because the 8B model cannot fit in FP32 on the available 32 GB V100. The
cache tensor gates remain strict; the logit KL tolerance accounts only for the
expected difference between full-sequence and split-cache FP16 reductions.

There is no Writer, Reader, LoRA, optimizer, or parameter update.

```bash
mkdir -p logs
bash run_with_cuda_resume.sh smoke 2>&1 | tee -a logs/smoke.log
bash run_with_cuda_resume.sh development > logs/pipeline.log 2>&1
```

Per-sample records make the audit resumable after a CUDA interruption.
