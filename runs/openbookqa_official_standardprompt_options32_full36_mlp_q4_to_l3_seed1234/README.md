# OpenBookQA reverse Full36-MLP KV translation

This experiment mirrors the clean forward OpenBookQA protocol in the reverse direction:

- sender: Qwen3-4B (36 layers, 8 KV heads, head dimension 128);
- receiver: Llama3.2-3B-Instruct (28 layers, 8 KV heads, head dimension 128);
- official OpenBookQA Main train/dev/test splits, sampled as 1024/128/128 with seed 1234;
- standard `Question -> Options -> Answer:` prompt;
- sender routing candidates are restricted to Options, with 32 region-balanced tokens;
- receiver preserves its native Question cache and reads translated Option KV before `Answer:`;
- translator is Full36-MLP: every target layer and head has an independent MLP, with separate bias-free K and V parameters;
- Stage A trains representation reconstruction from scratch;
- Stage B compares shared-parameter fine-tuning with a frozen Stage-A base plus a rank-64 receiver-space residual adapter.

Run smoke:

```bash
/home/yezhe/data/miniconda3/envs/attnkv/bin/python -u run_pipeline.py --mode smoke all
```

Run study:

```bash
/home/yezhe/data/miniconda3/envs/attnkv/bin/python -u run_pipeline.py --mode study all
```

The final summary is written to `runs/<mode>/results/final_metrics.json`.
