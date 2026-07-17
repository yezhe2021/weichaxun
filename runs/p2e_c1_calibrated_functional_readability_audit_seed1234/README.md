# P2-E-C1 Calibrated Functional Readability Audit

This revision fixes the invalid positive control in C0.

The Llama final-evidence-hidden stage directly reuses and freezes the successful
Experiment-A `raw_evidence_attention_layer_28` checkpoint. It performs inference
only and must recover the known final-hidden upper bound on the C1 test split.

Representations that are not input-compatible with that checkpoint are trained
normally on the full Experiment-A split:

```text
448 train pairs / 64 validation pairs / 64 test pairs
```

KV probes preserve the token axis. For each token, all KV heads are concatenated
before independent per-layer K/V projection; token attention is then performed
without flattening heads into the token axis. All newly trained probes use up to
30 epochs, validation early stopping with patience 6, soft layer weighting, and
the same seed and split.

The audited chain remains:

```text
reused frozen final-hidden probe (positive control)
-> newly trained raw last/all-layer Native-KV probes
-> newly trained task_only/shared_span_relation Writer-KV probes
-> newly trained Reader readout/cumulative delta/final hidden probes
-> first-token logits
-> free-running generation
```

Controls include paired counterfactual state swap, answer-disjoint shuffled
state, K-correct/V-mismatched memory, zero state, and Reader-off where defined.

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
