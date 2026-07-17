# Experiment A: Prefill-State Answerability

This experiment tests whether a frozen Llama-3.2-3B-Instruct prefill already
contains a recoverable city answer. It does not run free-generating Llama during
probe training or evaluation.

One frozen forward appends 16 identical, non-special, fixed-token summary slots
after the calibrated P2-E prompt and caches:

- the state immediately before the summary slots at four normalized depths;
- all 16 summary-slot states at those depths;
- final-layer hidden states for the actual Evidence A/B tokens.

The following probes are trained while Llama remains frozen:

- `end_linear` at each cached depth;
- `summary_8_linear` and `summary_16_linear`;
- `summary_8_attention` and `summary_16_attention`;
- `raw_evidence_attention` on the current evidence-token transfer object.

Training uses only correct A+B memories. Evaluation additionally includes
question-only, A-only, B-only, answer-masked, shuffled-state and paired
counterfactual state-swap controls. The data remains grouped by counterfactual
pair when creating the train/validation split.

Interpretation boundary:

- strong summary probes show that frozen prefill can form an answer-bearing
  aggregate state;
- only a strong raw-evidence probe directly supports the current transmitted
  evidence-token object;
- summary success with raw-evidence failure motivates transmitting learned
  summary slots instead of forcing a Writer to reconstruct receiver-native KV.

Run:

```bash
bash run_all.sh all
bash run_all.sh status
```

Primary output:

```text
probe_results/SUCCESS.json
probe_results/probe_comparison.csv
```
