# P2-Hotpot: Frozen W8 Canonical Writer to Frozen R4 Reader

Zero-shot domain-transfer audit on 64 fixed HotpotQA distractor-dev examples. The data builder exposes only gold supporting sentences to isolate evidence communication and multi-hop answering from retrieval.

All existing modules remain frozen:

- Qwen3-8B sender backbone
- P2-I-W token-preserving Canonical Writer
- Qwen3-4B receiver backbone
- P2-I-R Qwen3-4B Canonical Reader

The evaluation reports open-answer HotpotQA-style EM/F1 for question-only, direct full-text Qwen3-4B, direct full-text Qwen3-8B, and Canonical correct/shuffled/zero/reader-off/KV-mismatch/token-permutation conditions. It does not use the synthetic 40-city answer vocabulary and does not stop on metric thresholds.

```bash
bash /home/yezhe/伪查询/runs/p2hotpot_w8_to_r4_zeroshot_seed1234/run_all.sh all
```
