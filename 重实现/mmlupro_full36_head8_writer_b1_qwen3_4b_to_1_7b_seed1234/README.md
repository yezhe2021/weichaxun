# Qwen3-4B→1.7B Full-36 + Head-8 Writer

This experiment extends the earlier layer-only reverse Full-36 Writer with
explicit per-target-head mixing. It reuses the same MMLU-Pro manifests and
native Full-KV caches: `target4` is the 36-layer Sender and `source17` is the
28-layer frozen Receiver target.

Each of the 28 target layers owns independent K/V depth maps over all 36
source layers. Each target layer and each of its eight KV heads then owns an
independent K/V head map over all eight intermediate heads. Every linear map
uses `bias=False`; tokens are never mixed.

Stage A reconstructs Native 1.7B KV. Stage B optimizes only final-position
full-vocabulary KL against the Native 1.7B trajectory.
