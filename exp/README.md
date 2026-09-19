# Experiments

This is the canonical location for new experiment uploads.

Directory convention:

```text
exp/<short-experiment-name>/
```

Each experiment should keep only reproducibility artifacts:

- source code and configuration;
- README and launch commands;
- logs and final result summaries;
- lightweight audit files needed to interpret the run.

Do not commit model weights, checkpoints, generated KV caches, teacher caches,
pair caches, Python bytecode, or other large intermediate artifacts.

New experiment names should be short, ASCII-only, and use lowercase snake_case
where practical so links remain easy to share.
