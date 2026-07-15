# P2-C3 Structure-Preserving Writer Ablations

This experiment keeps the Qwen3-8B receiver backbone and the trained P2-A2
Query-only Reader strictly frozen. Only the 1.7B-to-8B Writer is optimized.
It reuses the exact P2-C1/P2-C2 sender and native-teacher KV caches.

## Variants

| Variant | Shared K/V main routing | Token/relation losses |
| --- | --- | --- |
| `task_only` | No | No |
| `shared_routing` | Yes | No |
| `binding_relation` | No | Yes |
| `shared_routing_relation` | Yes | Yes |

The 2x2 design separates the effect of route sharing from the effect of
structure regularization. All variants use per-head rank-32 residual adapters,
top-6 layer routing, the same identity-like initialization, 512 pairs, three
epochs, seed 1234, and the same free-running evaluation conditions.

Shared-routing variants use one main layer route and one main head map for K
and V. K/V may learn only bounded residual logits. Sparse layer selection uses
the shared main route, so K and V always retain the same top-k layer support.

## Objective

The primary objective is answer generation: base/counterfactual NLL, answer
exchange margin, and correct-memory ranking against shuffled or value-mismatched
memory. Route, readout, and target-attention alignment are small auxiliaries.

Structure variants additionally preserve:

- token-level K/V cross-token binding;
- K-token and V-token cosine relation graphs;
- query-conditioned per-token contribution graphs formed from Reader routes
  and V content.

Relation sampling covers the whole evidence span and always includes answer
tokens, with at most 64 positions per layer to bound CPU cost.

## Run

```bash
bash run_all.sh all
bash run_all.sh status
```

The four variants run sequentially to avoid loading multiple frozen 8B models
at once. Completed variants are skipped safely on restart.
