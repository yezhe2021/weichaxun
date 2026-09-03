# Qwen3-4B → Qwen3-1.7B reverse diagnosis (seed 1234)

This directory contains the implementation and lightweight JSON outputs for the reverse-direction diagnosis experiment. Model weights, checkpoints, and the 137 GB native-KV cache are intentionally excluded.

## Rapid feasibility validation

The original long pipeline was stopped after a valid 150-update checkpoint had been saved for the continued fixed-depth branch. The rapid run reused that checkpoint and the existing cache, trained only the missing learnable-depth branch for 150 CE updates, and evaluated 32 held-out examples under correct and shuffled cache controls.

| Branch | Correct EM | Correct F1 | Shuffled EM | Shuffled F1 | Correct NLL | Shuffled NLL | Correct − shuffled F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 continued fixed depth | 0.0625 | 0.146875 | 0.1250 | 0.211458 | 3.093998 | 3.413048 | -0.064583 |
| R2 learnable depth | 0.09375 | 0.131250 | 0.0625 | 0.143750 | 2.646301 | 3.466774 | -0.012500 |

The learnable-depth branch shows a clear correct-cache advantage in answer NLL, but neither branch shows a positive correct-versus-shuffled generation-F1 advantage on this 32-example rapid test. This is evidence of a probability-level signal without reliable generation-level feasibility under the tested budget; it is not a final performance estimate.

Reference 1.7B results on 128 examples:

- Question only: EM 0.085938, F1 0.096503
- Full context text: EM 0.195312, F1 0.226780
- Native cache: EM 0.195312, F1 0.226780

## Reproduction notes

- Full configuration: `config.json`
- Rapid configuration: `config.quick.json`
- Rapid entry point: `quick_pipeline.py`
- Main rapid summary: `artifacts/development/quick_validation/summary.json`
- Runtime log: `logs/quick_validation.log`
- The source manifests are reused from the earlier forward and 8B→4B experiments specified in `config.json`.
