# MMLU-Pro Options-first Full-KV Trajectory Alignment

This independent experiment tests whether a question-independent Qwen3-1.7B Options cache can be translated into Qwen3-4B native Full KV well enough to reproduce the frozen 4B Receiver's execution on a later Question.

```text
original-order MMLU-Pro Options only
  -> frozen Qwen3-1.7B
  -> 28-layer full pre-RoPE K / native V
  -> D0 / D1 / D2 Writer
  -> 36-layer 4B-format full pre-RoPE K / native V
  -> frozen Qwen3-4B + Question + Answer:
```

The Writer never sees the Question. MMLU-Pro `answer`, `answer_index`, and `cot_content` never enter Stage A or Stage B. Gold labels are used only by protocol feasibility measurement and final evaluation.

## Locked protocol

- Keep the original MMLU-Pro option order; do not permute labels.
- Assert `tokenize(prefix + suffix) == tokenize(prefix) + tokenize(suffix)` for both models.
- MMLU-Pro has a variable number of choices. Score `A` through the sample's final valid label (up to `J`) and assert every label token is unique.
- Capture `k_norm(k_proj(hidden))` before RoPE and native `v_proj(hidden)`.
- Apply the frozen 4B Receiver's own RoPE when building `DynamicCache`.
- D0/D1/D2 use independent, bias-free K and V maps for every target layer, without token or head mixing.
- RMS calibration uses derived training samples only and is frozen.

## Losses

Stage A aligns all 36 target K/V layers with layer-normalized NMSE and cosine losses.

Stage B uses the same frozen Qwen3-4B as teacher and student Receiver. The only difference is Native versus translated prefix KV:

```text
lambda_kv * 36-layer KV reconstruction
+ lambda_hidden * 36-layer/all-suffix-token hidden trajectory alignment
+ lambda_kl * suffix-token LM distribution KL
```

Forward hooks capture the exact output of each of the 36 decoder blocks. They are supervised separately and averaged; the embedding state and final LM-head-only projection are excluded. The teacher runs under `torch.no_grad()`. The student Receiver remains frozen but must run with autograd so gradients reach translated KV and the Writer.

## Data note

The ModelScope copy contains an official 70-row validation file and a 12,032-row test file, but no train split. The default MVP configuration creates explicitly marked, disjoint derived train/validation/test subsets from the official test file. These results are not directly comparable to the MMLU-Pro leaderboard. Switch `data.split_strategy` to `heldout_category` and supply disjoint category lists for the formal generalization experiment.

## Explicit execution order

Nothing starts automatically:

```bash
python -u run_unit_tests.py
python -u run_pipeline.py --config config.json prepare
python -u run_pipeline.py --config config.json audit
python -u run_pipeline.py --config config.json cache --split train --family both
python -u run_pipeline.py --config config.json cache --split validation --family both
python -u run_pipeline.py --config config.json cache --split test --family both
python -u run_pipeline.py --config config.json calibrate

python -u run_pipeline.py --config config.json train --writer d2 --stage a
python -u run_pipeline.py --config config.json gradient_audit --writer d2 --checkpoint checkpoints/quick/d2/stage_a/best.pt
python -u run_pipeline.py --config config.json overfit --writer d2 --stage both

python -u run_pipeline.py --config config.json train --writer d0 --stage both
python -u run_pipeline.py --config config.json train --writer d1 --stage both
python -u run_pipeline.py --config config.json train --writer d2 --stage b
python -u run_pipeline.py --config config.json evaluate --writer d0 --stage both
python -u run_pipeline.py --config config.json evaluate --writer d1 --stage both
python -u run_pipeline.py --config config.json evaluate --writer d2 --stage both
```

The protocol audit is the only hard correctness gate. Overfit and training losses are evidence and never block later stages.

## Evaluation conditions

- 4B standard Question-first full text
- 4B Options-first full text
- 4B Native Full KV
- 1.7B standard and Options-first full text
- Question only
- D0/D1/D2 Stage A and Stage B
- true zero-valued cache
- exact-prefix-length cross-sample shuffled cache

Report Accuracy, Native Agreement, A-J logit cosine/KL, Correct-minus-Zero, Correct-minus-Shuffled, and Stage-B-minus-Stage-A.

## Storage

Full source/target KV caches and checkpoints stay on the server and are ignored by Git. Machine-readable manifests, summaries, trajectories, per-sample A-J outputs, and logs may be published. Caches can be deleted after final results are saved.
