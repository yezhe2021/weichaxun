# Full-28 Choice-Mass / Functional-Distance Audit

Read-only audit of the existing 1.7B→4B Full-28 Stage-A and B1 checkpoints. It performs no training,
checkpoint selection, or answer-conditioned optimization.

For Native4 on train/validation/test it records A–J probability mass, conditional entropy, and top1–top2
margin. For each Writer checkpoint it records full-vocabulary KL, choice-only KL, centered choice-logit MSE,
top1 agreement, and an exact three-part KL decomposition (mass, weighted choice, non-choice conditional).

Run:

```bash
bash launch_experiment.sh
tail -f logs/pipeline.log
```

Outputs are under `artifacts/formal/`; per-sample JSONL preserves auditability while `summary.json` contains
quantiles, contribution shares, and margin-bucket analyses.
