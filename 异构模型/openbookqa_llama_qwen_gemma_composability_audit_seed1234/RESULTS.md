# Llama3.2-3B → Qwen3-4B Hub → Gemma3-4B results

Status: completed on 128 official OpenBookQA Main test examples. All four checkpoints were
frozen; no training occurred in this experiment.

## Upstream Qwen behavior

| Llama → Qwen state | Qwen accuracy | K cosine to native Qwen | V cosine to native Qwen |
|---|---:|---:|---:|
| Stage-A | 46.88% | 0.95248 | 0.80064 |
| Residual Stage-B | 74.22% | 0.95058 | 0.79931 |

Residual Stage-B improves Qwen accuracy by 27.34 percentage points while preserving most of
the Stage-A representation geometry.

## Composition through frozen Qwen → Gemma writers

| Fixed downstream writer | Qwen Hub input | Gemma accuracy | Agreement with Gemma Oracle32 | Choice-KL to Oracle32 |
|---|---|---:|---:|---:|
| Downstream Stage-A | Qwen native bridge | 70.31% | 75.78% | 0.76302 |
| Downstream Stage-A | Llama Stage-A | 61.72% | 65.62% | 0.86757 |
| Downstream Stage-A | Llama Residual Stage-B | **61.72%** | 65.62% | 0.87137 |
| Downstream Residual | Qwen native bridge | 71.09% | 77.34% | 0.49633 |
| Downstream Residual | Llama Stage-A | 61.72% | 64.84% | 0.67440 |
| Downstream Residual | Llama Residual Stage-B | **62.50%** | 64.06% | 0.67725 |

Gemma native Oracle32 at the same Llama-selected raw-text anchors is 71.09% (91/128).

## Paired correctness

- With downstream Stage-A, upstream Stage-A and Stage-B produce exactly the same correctness
  partition: 79 both correct and 49 both wrong.
- With downstream Residual, there are 78 both correct, 1 Stage-A-only, 2 Stage-B-only, and 47
  both wrong, for a net Stage-B gain of one example.

## Interpretation

The Llama residual Stage-B state remains consumable by an independently trained
Qwen→Gemma writer: it does not cause a downstream accuracy collapse, and it gives a small
gain with the downstream residual branch. However, the large Qwen-local functional gain
(`+27.34` points) does not transfer through composition (`+0.00` or `+0.78` points).

This distinguishes two properties:

1. **Canonical compatibility is largely preserved**: the state can traverse the second writer.
2. **Functional improvement is receiver-specific**: most Stage-B gains are calibrated to Qwen
   readout and are not encoded as universally useful Hub information for Gemma.

The 8.59-point gap between the best composed state (62.50%) and the Qwen-native bridge
(71.09%) shows that the first translation remains the dominant compositional bottleneck.

Machine-readable outputs and per-sample predictions are under `runs/study/results/`.
Checkpoints and intermediate caches are intentionally excluded.
