# P2-E-C0 Quick Segmented Functional Readability Audit

This is the 64/16/16 quick directional audit. It reuses the Experiment-A split,
existing Llama hidden/KV caches, the two frozen P2-E Writers, and the frozen
P2-A2 Query-only Reader.

The audited path is:

```text
Llama final evidence hidden
-> raw last/all-layer Native KV
-> task_only or shared_span_relation Writer KV
-> frozen Reader all-layer readout
-> cumulative injected delta / Qwen final hidden
-> first-token candidate logits
-> free-running generation
```

All functional probes use latent width 64, soft layer weighting, the same
classifier capacity, six fixed epochs, seed 1234, and pair-level base/CF data.
The test controls include counterfactual state swap, answer-disjoint shuffled
state, K-correct/V-mismatched memory, zero state, and Reader-off where defined.

The quick audit labels a drop only when paired consistency decreases by at
least 20 percentage points. With 16 test pairs this is directional evidence,
not a statistical-significance claim.

```bash
bash run_all.sh all
bash run_all.sh status
```

Primary outputs:

```text
comparison/unified_stage_table.csv
comparison/first_large_drop.json
comparison/SUCCESS.json
```
