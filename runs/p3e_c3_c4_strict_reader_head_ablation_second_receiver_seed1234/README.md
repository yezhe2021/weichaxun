# P3-E C3/C4 Strict Reader and Second Receiver Pipeline

This pipeline never updates the frozen C2 Qwen3-8B Head-Structured Writer.

Execution order:

1. C3-A fully-random Qwen3-4B Reader, seeds 1234 and 2345.
2. C3-A weak-pair-prior Qwen3-4B Reader, seeds 1234 and 2345.
3. C3-B paired Canonical-head interventions on both fully-random Readers.
4. C4 Qwen3.5-4B Reader onboarding through its eight genuine full-attention layers, seeds 1234 and 2345.

All Reader backbones and native output projections remain frozen. C3-A loads neither the P3-E-B Native Reader nor the C1 Reader. C3-B freezes both Writer and Reader. C4 maps the 16 Canonical memory layer groups into eight fixed adjacent pairs and preserves Qwen3.5's native Query gate and frozen `o_proj`.

The stages execute unconditionally and do not use validation scores for early stopping or routing decisions.

```bash
bash /home/yezhe/伪查询/runs/p3e_c3_c4_strict_reader_head_ablation_second_receiver_seed1234/run_all.sh all
```
