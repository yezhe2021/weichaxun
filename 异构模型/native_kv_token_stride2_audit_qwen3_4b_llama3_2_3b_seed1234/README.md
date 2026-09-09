# Native KV token stride-2 audit

No parameters are trained. For Qwen3-4B and Llama-3.2-3B, the same 128 MMLU-Pro test
examples are evaluated under three cache conditions:

- `native_full`: all native context K/V tokens.
- `keep_even`: retain context tokens 0, 2, 4, ... and synchronously delete the rest from K/V.
- `keep_odd`: retain context tokens 1, 3, 5, ... and synchronously delete the rest from K/V.

Retained keys keep their original RoPE position IDs, and the question suffix starts at the
original full-context length. This prevents token deletion from being confounded with compact
position reindexing. Metrics are full-vocabulary KL, choice-only KL, Native agreement and
MMLU-Pro Accuracy. Existing pre-RoPE Native KV is read without creating a new cache.

```bash
cd /home/yezhe/异构模型/native_kv_token_stride2_audit_qwen3_4b_llama3_2_3b_seed1234
bash launch.sh
tail -f logs/pipeline.log
cat runs/formal/status.json
```
