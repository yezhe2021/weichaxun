# P3-E-H Evidence Assimilation Reader V2

P3-E-H keeps the Qwen3-8B Sender, C2 Writer, Canonical cache, Qwen3-4B
backbone, native attention/o_proj, and complete C1 retrieval path frozen.

The first eight C1 injection layers are unchanged. At the last eight injection
layers, a Receiver-specific adapter combines the raw Receiver residual state
with the ungated C1 external evidence residual:

```text
u = SiLU(W_h RMSNorm(h) + W_z RMSNorm(z))
assimilation = U(u)
delta = old_gate * z + beta * assimilation
beta = 0.1 * tanh(beta_logit)
```

`U` is zero-initialized. `beta` starts at 0.01 rather than zero: initializing
both `U` and `beta` to zero would make every gradient in the new branch zero.
Because `U=0`, the initial Reader output still exactly reduces to C1.

## Full-text behavior teacher

The frozen Qwen3-4B teacher reads Question plus complete gold Evidence. For each
of 512 training examples, only the gold-answer prediction positions are cached:
top-256 token IDs and raw logits. Inputs are not truncated. Student and teacher
positions are aligned by answer-token order rather than absolute prompt index.

## Training

- Smoke16: 5 epochs.
- Formal512: independently initialized from C1, 5 epochs.
- Loss:
  `L_answer + 0.5*relu(0.5+NLL_correct-NLL_shuffled) + 0.5*KL_T=2`.
- Validation64 is evaluated once after training and is not used for selection.
- Only the eight late-layer Assimilation modules are optimized.

## Evaluation

Conditions are `question_only`, `current_c1_reader`, `assimilation_reader_v2`,
`hard_shuffled_assimilation`, `oracle_support_assimilation`, and `reader_off`.
Outputs include automatic HotpotQA metrics, bridge/comparison splits, memory
switch behavior, old gates, beta values, assimilation RMS/cosine diagnostics,
per-example generations, and a blinded manual C/P/W worksheet.

```bash
bash /home/yezhe/伪查询/runs/p3e_h_evidence_assimilation_reader_v2_seed1234/launch.sh
tail -f /home/yezhe/伪查询/runs/p3e_h_evidence_assimilation_reader_v2_seed1234/p3e_h_run.log
```
