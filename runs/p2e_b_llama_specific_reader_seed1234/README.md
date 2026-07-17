# Experiment B: Llama-Specific Reader

This experiment removes the Writer and trains a new Reader that directly reads
the existing evidence-token-only Llama-3.2-3B Native-KV cache. The Qwen3-8B
receiver backbone is strictly frozen.

Variants:

- `minimal_reader`: deterministic normalized-depth layer matching and fixed
  near-identity KV-head mapping; trains only per-layer Query adapters, Output
  adapters, and gates.
- `routed_reader`: additionally learns shared K/V top-2 global layer routing and
  shared K/V head mapping. It attends to each selected Llama layer separately
  and mixes readouts, so K and V always come from the same source layer/head
  route.

Training is driven by base/counterfactual answer NLL, answer-swap margin, and
correct-vs-shuffled/mismatched memory ranking. Evaluation is free-running and
includes full text, the trained Qwen Native-KV Reader upper bound, correct
Llama-KV, shuffled, mismatched, zero, and Reader-off conditions.

This is a sender-family-specific Reader for raw Llama KV. It does not establish
a receiver-independent Canonical Evidence-KV space.

```bash
bash run_all.sh all
bash run_all.sh status
```
