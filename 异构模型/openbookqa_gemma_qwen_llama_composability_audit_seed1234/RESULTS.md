# Results

Status: completed on 128 official OpenBookQA Main test examples (seed 1234).

## Question tested

Does the frozen Gemma3-4B → Qwen3-4B residual Stage-B state preserve its usefulness as a
canonical Qwen-space state when consumed by an independently trained Qwen3-4B →
Llama3.2-3B writer?

## Upstream Qwen result

| Gemma → Qwen state | Qwen accuracy | K cosine to native Qwen | V cosine to native Qwen |
|---|---:|---:|---:|
| Stage-A | 40.62% | 0.950766 | 0.778630 |
| Residual Stage-B | 74.22% | 0.949200 | 0.777011 |

Stage-B improves Qwen accuracy by 33.59 percentage points while changing K/V cosine only
slightly.

## Composition through frozen Qwen → Llama writers

| Fixed reverse writer | Input Qwen-space state | Llama accuracy | Agreement with Llama native Oracle32 | Choice-KL to Oracle32 |
|---|---|---:|---:|---:|
| Reverse Stage-A | Qwen native bridge | 63.28% | 72.66% | 0.20436 |
| Reverse Stage-A | Gemma Stage-A | 57.81% | 75.00% | 0.21726 |
| Reverse Stage-A | Gemma Residual Stage-B | **58.59%** | 74.22% | **0.21607** |
| Reverse Residual | Qwen native bridge | 62.50% | 70.31% | 0.20084 |
| Reverse Residual | Gemma Stage-A | 53.91% | 75.78% | 0.19355 |
| Reverse Residual | Gemma Residual Stage-B | **55.47%** | **76.56%** | **0.19272** |

The Llama-native Oracle32 at the same Gemma-selected raw-text anchors is 65.62% (84/128).

## Paired correctness

For the fixed reverse Stage-A writer, Stage-A versus Stage-B has 74 both-correct, 0
Stage-A-only, 1 Stage-B-only, and 53 both-wrong examples. For the fixed reverse residual
writer, the counts are 69, 0, 2, and 57 respectively.

## Interpretation

The residual Stage-B state preserves composability in this test: its large upstream Qwen
functional gain survives an independently trained reverse writer, with no loss in downstream
accuracy and a small improvement under both reverse-writer variants. The downstream K/V
cosines change by only about 0.0003--0.0005. This is direct evidence for compatibility on this
model pair and protocol, though it is not yet evidence of universal Hub compatibility across
unseen receiver models or tasks.

Machine-readable outputs are under `runs/study/results/` on the experiment server.

## Greedy generation audit

The same frozen checkpoints were evaluated with actual greedy decoding for up to 12 new
tokens after the receiver-native `Answer:` suffix. No retraining was performed.

| Condition | Valid A-D first token | Parsed generation accuracy |
|---|---:|---:|
| Llama native Oracle32 | 100% | 64.84% (83/128) |
| Qwen-native bridge → reverse Stage-A | 100% | 62.50% (80/128) |
| Gemma Stage-A → reverse Stage-A | 100% | 57.81% (74/128) |
| Gemma Residual Stage-B → reverse Stage-A | 100% | **58.59% (75/128)** |
| Gemma Residual Stage-B → reverse Residual | 100% | 55.47% (71/128) |

Every condition generated a valid A-D token first on all 128 examples. The main Stage-B
composition therefore preserves both answer formatting and the small downstream improvement
observed in the choice-logit audit. Full decoded continuations are stored in
`runs/study/generation/per_sample_generations.jsonl`.
