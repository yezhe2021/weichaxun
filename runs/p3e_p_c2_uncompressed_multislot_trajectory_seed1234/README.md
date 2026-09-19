# P3-E-P-C2 Uncompressed Multi-Slot Trajectory Reader

This experiment preserves the original P3-E-P C1 result as the pooled-memory
baseline and adds the corrected full-KV capacity test.

C2 uses the same frozen Qwen3-4B Teacher/Receiver, question-conditioned Native
KV `[16,T,8,128]`, 512/64 split, 16 layers, seed, optimizer, eight epochs, and
`answer + 0.5 * trajectory-state` loss as C1.

At each layer, the current Receiver state initializes eight independent 256D
slots. Each slot performs two rounds of head-preserving cross-attention over
the complete Native K/V. All eight slots remain present after decoding. There
is no slot pooling before trajectory correction. The current Receiver hidden
state then queries those eight slots, and that state-conditioned result enters
the shared trajectory corrector to produce the 2560D residual update.

The final comparison retains Reader C, A1, original C1 pooled-128D, corrected
C2, C2 hard-shuffled, and full-evidence text.
