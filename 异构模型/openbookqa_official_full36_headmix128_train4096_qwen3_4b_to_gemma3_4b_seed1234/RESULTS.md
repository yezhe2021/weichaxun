# Qwen3-4B → Gemma3-4B reverse onboarding results

Status: completed on official OpenBookQA Main with deterministic seed-1234 splits
(`4096/128/128`).

## Final test results

| Condition | Accuracy | Agreement with Gemma Oracle32 | Choice-KL | K cosine | V cosine |
|---|---:|---:|---:|---:|---:|
| Gemma native Oracle32 | 71.09% (91/128) | 100% | 0 | 1 | 1 |
| Stage-A translator | 67.19% (86/128) | 71.88% | 0.78312 | 0.94946 | 0.81209 |
| Residual Stage-B | **70.31% (90/128)** | **74.22%** | **0.51324** | **0.94917** | **0.81170** |
| Shared Stage-B | 70.31% (90/128) | 73.44% | 0.61049 | 0.72597 | 0.74714 |

Residual Stage-B closes the translation gap from 3.91 to 0.78 percentage points while
preserving Stage-A representation geometry. It adds four correct answers over Stage-A and
loses none (`86` both correct, `0` Stage-A-only, `4` Residual-only, `38` both wrong).

Shared Stage-B reaches the same task accuracy but substantially damages canonical geometry,
especially K (`0.94946 → 0.72597`). Residual Stage-B is therefore the preferred checkpoint.

## Native and negative controls

| Baseline | Accuracy |
|---|---:|
| Qwen full native | 75.00% |
| Qwen native selected32 | 73.44% |
| Gemma full native | 75.78% |
| Gemma native Oracle32 at Qwen-selected anchors | 71.09% |
| Gemma self-selected32 | 70.31% |
| Gemma zero32 | 25.00% |
| Gemma no memory | 25.78% |
| Gemma shuffled Oracle32 | 26.56% |

The negative controls remain near four-choice chance, confirming that the translated memory
contains sample-specific information rather than only task or label priors.

## Architecture and training

- Source KV: Qwen `[36, 32, 8, 128]`.
- Target KV: Gemma `[34, 32, 4, 256]`.
- Bias-free Full36 depth MLP with independent target-layer and K/V parameters plus learned
  full-head mixing.
- Stage-A: 2048 optimizer steps, selected on official validation only.
- Stage-B: 1024 steps for frozen-base Residual64 and shared fine-tuning.
- Frozen Stage-A hash verification passed for the residual branch.

Machine-readable metrics and per-sample predictions are under `runs/study/results/`.
Checkpoints and intermediate KV caches are intentionally excluded from GitHub.
