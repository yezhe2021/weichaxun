# P0: Native-text compositional baselines

P0 establishes the lower bound, single-source controls, and full distributed-evidence upper bound before any Evidence-KV module is trained.

Each selected HotpotQA distractor example must have exactly two supporting titles. Their assignment to source A or B is deterministic under the run seed. Each source receives one gold supporting document and half of the distractor documents. `a_plus_b` contains both gold documents and all distractors, while preserving explicit source labels.

The four free-running conditions are:

- `question_only`: receiver sees no external evidence.
- `a_only`: receiver sees source A text only.
- `b_only`: receiver sees source B text only.
- `a_plus_b`: receiver sees both distributed sources.

No parameters are trained in P0. The default receiver is Qwen3-1.7B, loaded locally in FP16 on CUDA. Qwen thinking mode is disabled through the chat template when supported.

## Commands

```bash
cd /home/yezhe/伪查询

# Validate the data split, prompt rendering, and token budget without loading model weights.
MAX_SAMPLES=8 OUT=runs/composable_evidence_kv_p0_hotpotqa_seed1234/dry_run \
  bash runs/composable_evidence_kv_p0_hotpotqa_seed1234/run_all.sh dry-run

# Small end-to-end generation check.
MAX_SAMPLES=4 OUT=runs/composable_evidence_kv_p0_hotpotqa_seed1234/smoke \
  bash runs/composable_evidence_kv_p0_hotpotqa_seed1234/run_all.sh smoke

# Main P0 run.
MAX_SAMPLES=128 \
  bash runs/composable_evidence_kv_p0_hotpotqa_seed1234/run_all.sh run
```

## Outputs

- `manifest.jsonl`: deterministic A/B document allocation and dataset diagnostics.
- `per_sample_generation.jsonl`: prediction and metrics for every sample-condition pair.
- `condition_summary.csv`: EM/F1, truncation, token count, and latency by condition.
- `summary.json`: condition and paired compositional metrics.
- `SUCCESS.json`: completed-run marker and full resolved configuration.

The main P0 signal is `mean_ab_f1_gain_over_best_single`. `compositional_exact_match_rate` counts samples solved by A+B but by neither source alone. Literal answer-presence rates are reported because they expose samples where one source may reveal the answer without requiring the other hop.

P0 results are evidence of a usable data/model baseline only when A+B improves substantially over question-only and the best single source. Model outcomes are never used to filter the evaluation set.
