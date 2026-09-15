# Posterior-Question Token-Count Scaling Audit

Evaluation-only comparison of 32, 48, and 64 selected Options KV tokens for the
completed Llama-3.2-3B to Qwen3-4B posterior-Question Sender experiment.

The 48/64 arms reuse the exact validation-selected Full28-Diagonal Translator
parameters. The mapping is token-independent; this audit bypasses only the old
`T=32` input-shape assertion. No layer/head/K/V mapping is changed and no
training is performed.

For each token count, the audit evaluates both the position-matched Qwen native
KV oracle and translated Llama KV on the same 128 test samples. It also records
the actual number of unique source/target tokens and selected attention mass.

```bash
bash launch.sh
tail -f logs/pipeline.log
```

Final results: `runs/study/results/comparison.json`.
