# P2-F Qwen3.5-4B to Frozen Qwen3-8B Reader

This experiment changes only the sender family/version while keeping the
Qwen3-8B backbone and the P2-A2 Query-only Reader frozen.

Qwen3.5-4B is a hybrid model: layers 3, 7, 11, 15, 19, 23, 27, and 31 are
full-attention layers; the other 24 layers use linear attention. P2-F transfers
only genuine evidence-token pre-RoPE K/native V from those eight full-attention
layers. It does not reinterpret recurrent linear-attention state as token KV.

The Writer maps 8 layers, 4 KV heads, and head dimension 256 into the complete
Qwen3-8B Reader interface. No evidence-token compression is used.

Variants:

- `matched_task_only`: same task-dominant setup used for the Llama comparison.
- `reader_aligned`: adds stronger route KL, external-readout, and target-mass
  alignment against the frozen 8B Native-KV teacher on every training step.

Evaluation includes full text, 8B Native KV, deterministic minimal shape
alignment, Writer KV, counterfactual evidence, shuffled/mismatched/zero memory,
Reader-off, paired consistency, route/readout diagnostics, and per-sample output.

```bash
bash run_all.sh all
bash run_all.sh status
```

The result can only support claims about Qwen3.5 full-attention KV transfer.
It does not measure transmission of the 24 linear-attention recurrent states.
