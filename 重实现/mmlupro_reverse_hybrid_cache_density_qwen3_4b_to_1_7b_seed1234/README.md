# Reverse Hybrid Cache Density and Placement Pilot

Read-only 4B→1.7B reverse Full-36 B1 checkpoint audit on the fixed MMLU-Pro test-128 split.
It compares Native, 14/28 and 19/28 Writer-layer hybrids using interval/front/back placements, and Full Writer. K/V are always
replaced together. No Writer or Receiver parameter is trained or selected.

```bash
bash launch_experiment.sh
tail -f logs/pipeline.log
```

Formal outputs are written to `artifacts/formal/`.
