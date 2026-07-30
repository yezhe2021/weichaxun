# A1-v2 Qwen3-4B Reversible Canonical Roundtrip

Single-model hard-gated experiment:

`4B Native Sparse KV -> Writer4B -> Canonical -> Decoder4B -> Frozen R1 Sparse Reader`

It uses train-only fixed per-layer/per-head/per-dimension RMS scales, fixed
seeded K/V orthogonal bases, zero-preserving bias-free residual modules, and
no cross-model data or answer-CE training.

Stages are resumable and ordered as follows:

1. current-script Native Sparse Reader baseline;
2. train-only fixed statistics and canonical bases;
3. S0 untrained numerical, attention, and functional hard gate;
4. A1-v2 training for at most 500 optimizer updates;
5. final Native/Decoded/Shuffled/Zero/Bypass evaluation.

```bash
cd "$EXPERIMENT_ROOT/a1v2_qwen3_4b_reversible_canonical_roundtrip_seed1234"
tail -f logs/launcher.log
```
