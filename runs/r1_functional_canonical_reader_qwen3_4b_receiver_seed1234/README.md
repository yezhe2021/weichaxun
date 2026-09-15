# R1: Functional Canonical Reader Registration

This standalone stage reuses the exact R0 HotpotQA sample IDs and the frozen
P0-A shared Writer. It reconstructs missing original token positions from the
official raw examples and always captures states from the complete
`Context -> Question` sequence.

Stages:

1. prepare manifests with original selected/question positions;
2. extract 36-layer sparse native K/V and 8-layer frozen-Writer Canonical K/V
   for both Qwen3-4B and Qwen3-8B in 32-sample FP16 shards;
3. train and evaluate the R1-0 sparse-native q/o-LoRA oracle;
4. warm up the 8-to-36 depth mixer and residual MLP Translator by reconstruction;
5. train R1-1 on 4B Canonical using answer CE, reconstruction, and shuffled
   dependence margin;
6. freeze the Reader and evaluate R1-2 zero-shot on 8B Canonical.

No Selector, Writer update, sender ID, sender-specific Reader, residual
injection, k/v/MLP/lm-head LoRA, or full-parameter tuning is used.

```bash
cd /home/yezhe/可拔插/r1_functional_canonical_reader_qwen3_4b_receiver_seed1234
tail -f logs/pipeline.log
```
