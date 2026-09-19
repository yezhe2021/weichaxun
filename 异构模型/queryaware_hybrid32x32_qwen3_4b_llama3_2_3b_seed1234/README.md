# Query-aware Hybrid32x32

Frozen Qwen3-4B and Llama-3.2-3B each convert their own MMLU-Pro options-first Context KV into a fixed 65-entry interface:

`[token0, N0, S0, ..., N31, S31]`

Each normalized contiguous chunk retains one unified Native token index across all layers/heads. Query selection uses the frozen model's real RoPE/GQA Query-to-Context attention. The learned slot summarizes only the unselected tokens. Query, center, and deterministic-random selectors are trained separately with identical slot initialization and final-position full-vocabulary KL. Gold labels are evaluation-only.

Canonical positions are `0..64`; the receiver suffix starts at 65. The Query arm also receives a diagnostic original-position evaluation.

```bash
/home/yezhe/data/miniconda3/envs/attnkv/bin/python tests.py
/home/yezhe/data/miniconda3/envs/attnkv/bin/python -u run_pipeline.py --mode smoke
bash launch.sh
tail -f logs/pipeline.log
```
