# P1: Oracle Evidence Slots and External Reader

P1 isolates the receiver-side feasibility question: can a frozen receiver answer from fixed external evidence slots while its prompt contains only the question?

This stage does not use native KV. A frozen Qwen3-0.6B sender encodes the gold HotpotQA supporting sentences. Each source produces up to four masked sentence slots from one sender hidden layer. The trainable P1 adapter consists of independent K/V slot projections plus external cross-attention readers and learned gates at receiver layers 8, 16, and 24. The Qwen3-1.7B receiver backbone remains frozen.

The evaluation cache is reconstructed from the exact P0 manifest, preserving sample order and the A/B assignment. Training uses the HotpotQA train split; evaluation uses the P0 development set.

## Important semantics

- The receiver prompt always says that no external text evidence is provided.
- During training, evidence is injected only at answer-prediction positions.
- During free-running evaluation, evidence is read at the final prompt position for the first answer token and at every subsequent decode step.
- Sender slots remain fixed throughout decoding; the sender is not loaded during reader training or evaluation.
- Slot order has no positional encoding. `shuffled_a_plus_b` must produce the same generated tokens as `a_plus_b` apart from negligible numerical effects.

## Commands

```bash
cd /home/yezhe/伪查询

bash runs/composable_evidence_kv_p1_oracle_reader_seed1234/run_all.sh prepare-train
bash runs/composable_evidence_kv_p1_oracle_reader_seed1234/run_all.sh prepare-eval
bash runs/composable_evidence_kv_p1_oracle_reader_seed1234/run_all.sh train
bash runs/composable_evidence_kv_p1_oracle_reader_seed1234/run_all.sh eval
```

The stages are deliberately separate. `prepare-*` loads only the frozen sender and writes compact CPU caches. `train` and `eval` load only the frozen receiver plus the small adapter.

Useful smoke overrides:

```bash
TRAIN_SAMPLES=32 EVAL_SAMPLES=8 EPOCHS=1 \
  bash runs/composable_evidence_kv_p1_oracle_reader_seed1234/run_all.sh prepare-train
```

Use matching overrides for the remaining commands and distinct output/cache paths if preserving the formal run.

## Outputs

- `cache/train_oracle_slots.pt`, `cache/eval_oracle_slots.pt`: frozen oracle sentence slots.
- `train/checkpoint_latest.pt`: slot projections, readers, and gates only.
- `train/train_history.jsonl`: answer CE, mismatched contrast, gate, and gradient diagnostics.
- `eval/condition_summary.csv`: free-running EM/F1 for all controls.
- `eval/summary.json`: compositional, mismatch, permutation, and P0 gap-closure metrics.

P1 passes only if A+B improves over question-only and mismatched evidence, slot permutation is effectively invariant, and free-running performance closes a meaningful fraction of the P0 full-text gap. Teacher-forced CE is a training diagnostic, not the P1 success criterion.
