# Synchronized-Boundary Full28 Diagonal Translator

This experiment uses the deployable primary Oracle policy, Llama SyncMax with unique-fill, and keeps
the sample IDs, 32-entry budget, native Qwen token0, canonical positions, models, losses, and training
schedule fixed relative to the RawAnchor baseline. The sole methodological change is pairing Llama
and Qwen Native KV at the rightmost tokens of identical UTF-8 raw-prefix boundaries.

The pipeline materializes a local synchronized pair cache, then runs:

1. Full28 block-diagonal-head Stage A, 1024 optimizer steps at LR 1e-3, with independent bias-free K/V.
2. Functional validation for every 128-step checkpoint and validation-only checkpoint selection.
3. Stage B from Stage-A `best_accuracy`, 512 optimizer steps at LR 1e-4, pure final-position Choice-KL.
4. Test evaluation of both `best_accuracy` and `best_choice_kl`, plus RawAnchor/Sync comparison.

The Stage-B teacher is the Qwen Native synchronized-unit Oracle using the exact same 32 memories;
Qwen Full is not the Stage-B teacher. Token0 is excluded from Writer training and always bypassed as
Qwen Native. Run `python tests.py`, then `bash launch.sh`.
