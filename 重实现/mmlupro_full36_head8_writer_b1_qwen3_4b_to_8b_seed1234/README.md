# Qwen3-4B→8B Full-36 + Head-8 Writer

Reverse-direction counterpart of the 8B→4B experiment. It reuses the exact
MMLU-Pro manifests and native Full-KV caches, but swaps their roles:
`target4` is the Sender and `source8` is the frozen Receiver's Native target.

The Writer concatenates all 36 source layers, then applies independent
K/V maps for every target layer and independent same-head mixing matrices
for every target KV head. All linear layers use `bias=False`.

Stage A optimizes Native 8B KV reconstruction. Stage B optimizes only the
final-position full-vocabulary KL against the Native 8B trajectory.
