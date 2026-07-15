# P2-D Sender Answerability Demo

This demo tests whether the frozen Qwen3-1.7B sender can directly solve the
same paired counterfactual task when it sees the complete text evidence. It
does not use KV caches, Writer adapters, external Readers, or teacher forcing.

The default run evaluates 16 paired examples under:

- question only;
- Evidence A only;
- base and counterfactual Evidence B only;
- full text base and counterfactual evidence;
- current A with a mismatched B;
- fully shuffled evidence.

Negative examples are selected so that the current target person and target
organization do not occur in the replacement evidence and answer sets are
disjoint. Generation is greedy and free-running with the sender tokenizer and
chat template. The answer list is used only after generation for extraction.

```bash
bash run_demo.sh
```

Outputs are written to `demo_16_pairs/`:

- `SUCCESS.json`
- `condition_summary.csv`
- `per_sample_generation.jsonl`
