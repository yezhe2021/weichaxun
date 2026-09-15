# Token64 Full28-Diagonal Full Training

This is the normal-budget follow-up to the reduced Token64 pilot. It trains a
new Llama-3.2-3B to Qwen3-4B Full28-Diagonal Translator from scratch for 64
selected Options KV slots under the posterior-Question Sender protocol.

Budget:

- 1024 train / 128 validation / 128 test samples;
- Stage A: 1024 optimizer steps of KV reconstruction;
- Stage B: 512 optimizer steps of final-position A-J choice KL;
- batch size 8, validation every 128 steps;
- validation-only checkpoint selection.

The architecture keeps independent target-layer and K/V mappings, does not mix
heads or tokens, and uses bias-free linear layers. The only structural code
change relative to the earlier 32-token implementation is accepting a variable
token dimension.

Final evaluation automatically records paired correctness counts and sample IDs
for Translator64 vs Native64 Oracle, Translator64 vs Translator32, and Native64
Oracle vs Native32 Oracle. Successful completion is automatically appended to
`/home/yezhe/异构模型/EXPERIMENT_RESULTS.md`.

Use `bash launch_when_gpu_ready.sh` to wait for an idle GPU and launch once.
