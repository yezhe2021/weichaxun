# Native4 Channel Writer8 — Functional-only

This experiment removes KV, cosine, route, and attention-output losses from
backpropagation. They remain continuous diagnostics only.

Groups:

- F0: scale-only, untrained;
- F1: linear Writer initialized from scale-only;
- F2: linear Writer initialized from the previous KV-aligned Stage-A checkpoint.

F1 and F2 optimize only answer NLL, sparse Native-output distillation, and a
periodic correct-vs-shuffled dependence margin. The R1 4B Reader, Receiver, and
all q/o LoRA parameters remain frozen. The Writer receives evidence KV only and
never receives the question.

No hard pass/fail thresholds are used. This single-question-per-context round
does not claim multi-query memory reuse.

```bash
cd /home/yezhe/可拔插/native4_channel_writer8_functional_only_seed1234
tail -f logs/pipeline.log
```
