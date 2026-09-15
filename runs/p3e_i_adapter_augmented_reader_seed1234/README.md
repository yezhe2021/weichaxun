# P3-E-I Adapter-Augmented Reader

This experiment treats a small Receiver-specific LoRA set as part of the Reader
package. The Qwen3-8B Sender, C2 Writer, Canonical cache, complete C1 Headwise
Reader, and all original Qwen3-4B parameters remain frozen.

## QA-only V1

Only the last eight Qwen3-4B Decoder layers receive LoRA updates:

```text
layers 28-35
targets: q_proj, v_proj, o_proj, down_proj
rank: 8
alpha: 16
dropout: 0
```

LoRA A is Kaiming-initialized and LoRA B is zero-initialized, so the initial
Reader package is functionally identical to C1. LoRA parameters are float32;
their updates are cast back to the Receiver dtype.

The C1 Query Adapter, head routing, Canonical attention, scalar gates, and base
native `o_proj` weights are frozen. The additive `o_proj` LoRA affects both the
native self-attention output and C1 external headwise readout.

## Reader-off semantics

LoRA is formally part of the Reader package. `reader_off` disables both C1
external reading and all LoRA updates. It must match original `question_only`
token-for-token. Keeping LoRA active while disabling only external attention
would not be a valid Reader-off control.

## Training and evaluation

- Smoke16: five epochs.
- Formal512: independently initialized from the same C1 checkpoint, five epochs.
- Loss: `L_answer + 0.5*relu(0.5+NLL_correct-NLL_shuffled)`.
- Validation64 is evaluated only after training.
- Evidence Reconstruction is intentionally not run until QA-only results are
  examined.

Conditions are `question_only`, `current_c1_reader`,
`c1_reader_plus_lora_qa_only`, `hard_shuffled_lora`, `oracle_support_lora`, and
`reader_off`. Automatic metrics, bridge/comparison splits, switch behavior,
LoRA norm diagnostics, per-example outputs, and a blinded C/P/W worksheet are
saved.

```bash
bash /home/yezhe/伪查询/runs/p3e_i_adapter_augmented_reader_seed1234/launch.sh
tail -f /home/yezhe/伪查询/runs/p3e_i_adapter_augmented_reader_seed1234/p3e_i_run.log
```
