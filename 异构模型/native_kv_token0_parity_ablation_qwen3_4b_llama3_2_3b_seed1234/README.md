# Native KV token0 parity ablation

This no-training diagnostic isolates the effect of context token0 from even/odd stride-2
sampling. It uses exactly the same 128 MMLU-Pro test rows, prompts, cached pre-RoPE Native
KV and receiver injection protocol as the preceding stride-2 audit.

For Qwen3-4B and Llama-3.2-3B, it compares:

- `native_full`: all context KV tokens.
- `keep_even`: `0,2,4,6,...`.
- `keep_odd`: `1,3,5,7,...`.
- `keep_odd_plus_token0`: `0,1,3,5,7,...`.
- `keep_even_minus_token0`: `2,4,6,8,...`.

K and V always use identical retained indices. Retained keys keep their original RoPE
positions, and the receiver suffix starts at the original full-context length. Metrics are
full-vocabulary KL, choice-only KL, Native agreement and Accuracy.

```bash
cd /home/yezhe/异构模型/native_kv_token0_parity_ablation_qwen3_4b_llama3_2_3b_seed1234
bash launch.sh
tail -f logs/pipeline.log
cat runs/formal/status.json
```
