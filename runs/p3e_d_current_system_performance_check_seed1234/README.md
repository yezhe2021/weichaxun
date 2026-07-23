# P3-E-D Current-System Performance Check

This is an evaluation-only, 64-sample diagnostic of the current frozen single-Sender/single-Receiver system. It does not train or modify the Qwen3-8B Sender, C2 Learned Head-Structured Writer, P3-E-B Native Reader, C1 Canonical Reader, or Qwen3-4B Receiver.

## Conditions

- `question_only`
- `full_evidence_text`
- `supporting_text`
- `sender_summary_text`
- `native_headwise_kv`
- `learned_canonical_kv`
- `hard_shuffled_canonical_kv`
- `reader_off`

The full-text condition receives the exact complete Evidence A+B documents encoded by the Sender. The supporting-text condition receives only official supporting sentences. The Qwen3-8B summary is greedily generated with a maximum of 512 new tokens and may naturally contain the answer when it is supported by the evidence.

Receiver text is never silently truncated. Every prompt is checked against the model's actual context window and the run fails visibly if it cannot fit.

## Run

```bash
bash /home/yezhe/伪查询/runs/p3e_d_current_system_performance_check_seed1234/run_all.sh all
```

All expensive stages are sample-resumable. Final outputs are `summary.json`, `per_example.jsonl`, `prompts.jsonl`, `sender_summaries.jsonl`, checkpoint hashes in `manifest.json`, and stage-level timing records.
