# P3-A HotpotQA Canonical Evidence-KV Responsibility Decomposition

This experiment separates Writer information retention from Reader open-answer execution on real HotpotQA evidence.

- 512 fixed training examples and 500 fixed distractor-dev examples
- gold supporting sentences only
- frozen Qwen3-8B sender and frozen P2-I-W Writer
- controlled Readers for final hidden, raw shape-aligned KV, PCA KV, and Canonical KV
- a newly initialized Hotpot-specific Qwen3-4B Reader with a frozen backbone
- open-answer `FINAL:` parsing and HotpotQA-style EM/F1
- correct, shuffled, zero, Reader-off, and K/V-mismatch controls

The 32-example overfit stage is diagnostic only. The full pipeline continues regardless of metric thresholds and writes an automatic responsibility verdict at the end.

```bash
bash /home/yezhe/伪查询/runs/p3a_hotpot_canonical_responsibility_seed1234/run_all.sh all
```
