# MMLU-Pro Pure-Functional B1/B2 Full-KV Alignment

This independent follow-up preserves the original Full-KV protocol while making Stage A and Stage B causally orthogonal.

```text
Stage A: 1.7B Options Full KV -> D2 -> 4B-format KV
         loss = K/V representation reconstruction only

Stage B1: Native4 versus translated-cache final-position behavior
          loss = full-vocabulary KL(Teacher || Student) at Answer: only

Stage B2: Native4 versus translated-cache dense suffix behavior
          loss = mean full-vocabulary KL(Teacher || Student) after every suffix token
```

Stage B never reads MMLU gold labels and never includes KV reconstruction or hidden trajectory losses. KV and all 36 decoder-layer hidden metrics are computed under `torch.no_grad()` as diagnostics only.

## Locked protocol

- Sender sees the original-order Options only; Question arrives later at the frozen 4B Receiver.
- Full 28-layer Qwen3-1.7B pre-RoPE K/native V is translated to full 36-layer Qwen3-4B pre-RoPE K/native V.
- D2 has independent, bias-free K and V maps for every target layer, without token/head mixing.
- B1 and B2 instantiate fresh optimizers and independently reload the exact same Stage-A best checkpoint.
- Teacher is frozen and runs under `torch.no_grad()`; Student Receiver is frozen but its execution graph remains differentiable with respect to Writer KV.
- Training views physically remove `gold_index`, `gold_label`, `answer`, `answer_index`, and `cot_content`.
- Functional validation KL, never MMLU Accuracy, selects Stage-B checkpoints.

## B2 token convention

`all` compares the next-token distribution produced after each token in the Question/instruction/`Answer:` suffix. It includes the final answer-label distribution, but does not include prediction of the first Question token from Options KV alone.

## Reused assets

The paired Full-KV cache is reused read-only from the preceding experiment through `config.cache_dir`. Manifests and RMS calibration are regenerated deterministically in this experiment directory. Models, cache tensors, and checkpoints are not intended for Git.

## Checkpoints

```text
checkpoints/{overfit,quick}/d2/
  stage_a/best.pt
  stage_b_final/best.pt
  stage_b_all/best.pt
```

## Execution

```bash
bash run_all_experiments.sh
```

The runner executes, in order:

1. CPU unit tests, deterministic manifest preparation, protocol audit, RMS calibration, and real-GPU B1/B2 gradient-path tests.
2. D2 overfit Stage A (1,000 updates).
3. B1/B2 gradient audit from the same overfit Stage-A checkpoint.
4. Independent B1 and B2 overfit runs (2,000 updates each) and 16-sample evaluation.
5. Formal D2 Stage A (1,000 updates).
6. B1/B2 gradient audit from the same formal Stage-A checkpoint.
7. Independent B1 and B2 formal runs (2,000 updates each) and 128-sample evaluation.

## Evaluation

Formal and overfit summaries include:

- Native 4B, Question-only, Stage A, B1 Final-KL, and B2 All-token-KL.
- Correct, true-zero, and exact-prefix-length shuffled controls for B1/B2.
- Accuracy, Native Agreement, valid-choice logit cosine/KL.
- Diagnostic-only KV NMSE/cosine and 36-layer hidden NMSE/cosine for all three Writer checkpoints.
- Explicit shuffled-eligible counts.
