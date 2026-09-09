# Multi-View Functional Canonical Co-registration

First-shot minimum loop from the experiment specification:

- exact R1 HotpotQA sampling, truncated to 512/64/64 after an 8/4/4 smoke;
- full-context gold supporting-token Native pre-RoPE K/native V;
- all 36 layers, 8 KV heads, 128 dimensions, and dynamic token length;
- independent Qwen3-4B and Qwen3-8B Writers;
- source-blind temporary 4B and 8B Protocol Decoders;
- A1 self bootstrap, A2 four-path cross co-registration, and A3-4B functional
  calibration only;
- no Selector, compression, sender ID, Receiver LoRA, pair-specific branch, or
  Qwen3.5 participation.

Writer and Decoder projectors use `bias=False`; their zero-input behavior is
unit tested. A0 hard-stops on token, position, shape, mask, and shuffled-control
inconsistency.

```bash
cd "$EXPERIMENT_ROOT/a0_multiview_functional_canonical_qwen3_4b_8b_seed1234"
tail -f logs/launcher.log
```

`EXPERIMENT_ROOT` is the server directory that contains the pluggable
experiments. Development outputs intended for review are under
`artifacts/development/results/`; caches, logs, and model checkpoints are
deliberately excluded from Git.
