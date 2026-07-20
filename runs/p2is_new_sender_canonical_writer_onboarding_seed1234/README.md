# P2-I-S New Sender Canonical Writer Onboarding

P2-I-S freezes the existing Qwen3-8B public Writer and both P2-I-R Readers. It trains one Qwen3-4B
sender-specific Writer that emits the same variable-length `[T_E,256]` Canonical Evidence-KV for both frozen
Receivers.

The pipeline enforces strict Qwen3-4B/Qwen3-8B evidence-token ID equality, fits train-only ridge K/V maps,
performs token-level Canonical imitation, then calibrates one Writer against both frozen Readers. Receiver
gradients are taken with respect to detached memory leaves and propagated through the single cached Writer
forward only after both Reader losses are accumulated. Receiver and backbone parameters never receive
gradients or optimizer entries.

Four configurations are evaluated on the same held-out 64 pairs:

- `imitation_only`
- `q4_only` functional calibration, with zero-shot Qwen3.5 evaluation
- `dual_only` functional calibration without an ongoing Canonical anchor
- `full` Canonical anchor plus dual-Reader functional calibration

A 16-pair full-method overfit diagnostic is run first, but no metric threshold stops later stages. The main
deployment artifact is `FINAL_W4B_CHECKPOINT.pt`; ablation checkpoints are diagnostics and are never
Receiver-specific deployment Writers.

Free-running controls include old/new Sender replacement, base/CF memory swap, cross-sample shuffled memory,
current-K/other-V mismatch, zero, Reader-off, and synchronous token permutation.

```bash
bash run_all.sh all
bash run_all.sh status
```

Do not commit tensor caches, checkpoints, or logs. Commit scripts and JSON/JSONL summaries only.
