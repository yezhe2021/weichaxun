# P3-E-M Fresh Native Reader Q-Conditioning Diagnosis

This experiment removes Writer and Canonical KV and trains two independent,
isomorphic Native Headwise Readers from an identical random initialization:

- Reader A reads `KV(E)`.
- Reader B reads Evidence-token-only `KV(E|Q)`.

Both use the same 512 samples, five epochs, optimizer, learning rate, sample
order, seed, injection layers, parameter count, Receiver prompt, and decoding.
The frozen Qwen3-4B Receiver sees only the Question. No Question token KV is
transmitted.

The fixed 64-sample evaluation includes correct and hard-shuffled controls for
both Readers, question-only, reader-off, full evidence text, all generations,
and a manual C/P/W worksheet.
