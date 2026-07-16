# P2-E: Llama-3.2-3B Writer to a frozen Qwen3-8B Reader

This experiment changes only the sender family. The Qwen3-8B backbone and the
trained P2-A2 Query-only Reader are frozen. Only the receiver-specific Writer is
optimized.

Variants:

- `task_only`: independent K/V routing; generation, counterfactual swap, and
  wrong-memory ranking are primary. Transported route/readout alignment is a
  small periodic auxiliary.
- `shared_span_relation`: shared main K/V layer/head routing with small separate
  residuals, plus character-aligned span binding and relation preservation.

The Llama cache uses the calibrated `fewshot_join_reason` prompt, but only the
current sample's Evidence A/B token KV enters external memory. Cross-tokenizer
losses use overlap in canonical raw-evidence character coordinates rather than
assuming token identity.

```bash
bash run_all.sh all
bash run_all.sh status
```

This remains a Qwen3-8B Reader-specific per-layer KV Writer. It is not yet a
receiver-independent Canonical Evidence-KV representation.
