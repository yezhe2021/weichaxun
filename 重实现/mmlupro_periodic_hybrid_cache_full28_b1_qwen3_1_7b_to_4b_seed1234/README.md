# Periodic Hybrid Cache Pilot

Read-only 1.7B→4B Full-28 B1 checkpoint audit on the fixed MMLU-Pro test-128 split.
It compares Native, three every-third-layer hybrid offsets, and Full Writer. K/V are always
replaced together. No Writer or Receiver parameter is trained or selected.

```bash
bash launch_experiment.sh
tail -f logs/pipeline.log
```

Formal outputs are written to `artifacts/formal/`.
