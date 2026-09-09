# R0: Raw Native KV cross-model usability

Standalone experiment for Qwen3-4B/8B native KV and a frozen Qwen3-4B receiver.
It intentionally contains no Writer, Canonical space, Selector, Translator, sparse
layer scheme, distillation, alignment, contrastive, or dependence loss.

The pipeline:

1. samples HotpotQA without question-only filtering;
2. runs structural compatibility checks and a hard cache-replay equivalence gate;
3. extracts all 36 layers of FP16 native K/V from 4B and 8B;
4. trains independent `LoRA_self` and `LoRA_pair` readers on receiver `q_proj` and
   `o_proj` only;
5. evaluates all A--M controls and saves per-example generations and a 64-row
   manual C/P/W annotation template.

The built-in LoRA implementation is mathematically equivalent to rank-8,
alpha-16 LoRA and keeps FP32 trainable parameters. It avoids adding PEFT as an
environment dependency.

```bash
cd /home/yezhe/可拔插/r0_raw_native_kv_cross_model_qwen3_8b_to_4b_hotpotqa_seed1234
tail -f logs/pipeline.log
```

The launcher first performs a complete smoke run (8/4/4) and only then starts
the formal 1024/128/128 run. It waits for CUDA when the server GPU is temporarily
unavailable.
