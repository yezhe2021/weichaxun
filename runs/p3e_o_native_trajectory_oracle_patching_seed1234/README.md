# P3-E-O Native Trajectory Oracle Patching

This experiment trains no module. It runs two synchronized frozen Qwen3-4B
branches:

- Oracle: full gold evidence text plus the Target branch's generated prefix.
- Target: question-only plus its own generated prefix.

At Reader C's existing 16 layers, only the current token is patched:

- O1: native self-attention output after the frozen `o_proj`.
- O2: complete post-attention residual state.
- O3: complete decoder-block output.

For O2, the Target attention output is replaced by
`oracle_post_attention_state - target_layer_input`, so the actual residual sum
inside the unmodified decoder layer equals the Oracle post-attention state.

O1/O2/O3 align the shared answer suffix and generated prefix to the full-text
absolute positions. O3 is also run without position alignment as a sanity
control. The Oracle never receives the gold answer and always follows the
Target branch's free-running prefix; the next token is selected only from the
Target logits.
