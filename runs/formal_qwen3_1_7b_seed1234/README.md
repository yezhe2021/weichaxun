# Qwen3-1.7B translated-like KV experiment

This folder contains one complete repetition of the translated-like KV experiment
with Qwen3-1.7B.

- Training set: first 512 processed HotpotQA training examples
- Evaluation set: first 64 processed HotpotQA development examples
- Context limit: 256 tokens
- Generation length: 16 tokens
- Seed: 1234
- Device dtype: float16

Directory layout:

```text
controls/       deterministic controls
train/          five translator training runs
eval/           five checkpoint evaluations
summary/        merged diagnostics and completion manifest
run_all.sh      reproducible runner
```

`summary/SUCCESS.json` is written only after every required output is present and
non-empty. Checkpoints and full per-layer files are kept on the experiment server;
GitHub contains the runner, metadata, histories, aggregate tables, and sample-level
results.
