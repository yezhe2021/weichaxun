# P3-E-N3 Attention-vs-Readout Sufficiency Ablation

This diagnosis reuses the frozen Reader C states cached by P3-E-N2 and trains
three independent probes:

- `attention_only`: `[16,32,T]` Reader attention only.
- `readout_only`: `[16,2560]` post-`o_proj`, pre-gate Reader readout only.
- `attention_readout`: both inputs as the upper baseline.

All probes use the same 512 training samples, 64 validation samples, seed,
epochs, optimizer, and support-token/span/yes-no objectives. The readout-only
probe preserves the 16-layer axis with a layer Transformer and lets
position-specific token queries cross-attend those layer states; it never
averages the raw layer readouts before probing.

The Sender, Native KV, Reader, and Receiver are not loaded during probe
training. Correct, shuffled-source, and zero controls use the existing N2
cache.
