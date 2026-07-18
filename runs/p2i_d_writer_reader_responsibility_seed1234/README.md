# P2-I-D Writer-Reader Responsibility Diagnosis

P2-I-D does not modify the Canonical Evidence-KV interface. It reuses the final
P2-I Writer, Readers, Native-KV caches, and `[256,256]` Canonical caches to split
failure responsibility into three independently gated tests.

## Stages

1. **Frozen Writer probe**: a permutation-invariant attention probe is trained on
   448 pairs, selected on 64 held-out validation pairs, and tested on the original
   64-pair test set. A synthetic slot positive control must reach 95% before the
   real Writer result is considered valid.
2. **Slot diagnostics**: the frozen Writer is rerun over cached Native KV only.
   No sender model is loaded. The script measures slot cosine, effective rank,
   cross-sample variance, gate usage entropy, assignment entropy, and atom
   coverage without serializing the full slot-to-atom matrix.
3. **Free-slot Reader oracles**: eight fixed pairs receive independent trainable
   base/CF K/V slots. The receiver backbone is frozen; only the current Reader and
   free slots train. Qwen3-4B uses FP16 and Qwen3.5-4B uses FP32 because Qwen3.5
   FP16 backward was previously shown to produce non-finite Writer gradients.
4. **Real joint overfit**: runs only when the Writer probe and 4B Reader oracle
   pass. It trains the real Writer and one 4B Reader on the same eight pairs with
   only answer NLL and base/CF margin. A Reader-warmup rescue branch runs only if
   direct joint optimization fails.
5. **Responsibility table**: combines validity gates, free-running predictions,
   and control conditions into one verdict.

All stages save best/latest checkpoints, training curves, gradient diagnostics,
per-sample predictions, control summaries, and explicit success markers. Swap
and shuffled controls report both original-target accuracy and source-memory
accuracy; otherwise a correctly memory-controlled prediction could be counted as
an error.

## Run

```bash
cd /home/yezhe/伪查询
RUN=runs/p2i_d_writer_reader_responsibility_seed1234
bash "$RUN/run_all.sh" audit
bash "$RUN/run_all.sh" all
```

Detached execution:

```bash
cd /home/yezhe/伪查询
RUN=runs/p2i_d_writer_reader_responsibility_seed1234
mkdir -p "$RUN/logs"
nohup bash "$RUN/run_all.sh" wait-cuda >"$RUN/logs/p2id_run.log" 2>&1 &
echo $! >"$RUN/run.pid"
```

Status:

```bash
bash /home/yezhe/伪查询/runs/p2i_d_writer_reader_responsibility_seed1234/run_all.sh status
```
