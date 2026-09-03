# P3-E-N Native Oracle Bottleneck Decomposition

This experiment runs only the first diagnostic layer:

```text
Q + E -> frozen Qwen3-4B prefill
      -> Evidence-token-only 4B Native KV(E|Q)
      -> Fresh Native Headwise Reader C
      -> frozen Qwen3-4B receiving Question only
```

Reader C loads the exact shared random initialization used by P3-E-M Readers A
and B. It uses the same 512 samples, sample order, seed, architecture, parameter
count, AdamW settings, learning rate, answer-token mean NLL, five epochs, fixed
epoch-5 checkpoint, prompt, and decoding.

The fixed 64-sample evaluation compares correct and hard-shuffled 4B Native KV,
question-only, reader-off, and full text. P3-E-M Reader B is reused as the
heterogeneous 8B-to-4B reference. P3-E-N2 is not executed.
