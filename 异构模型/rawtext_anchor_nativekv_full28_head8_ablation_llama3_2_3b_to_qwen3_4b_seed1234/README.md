# Raw-text Anchor Native-KV Full28/Head8 Ablation

Llama-3.2-3B-Instruct to Qwen3-4B content Native-KV translation. The experiment reuses the exact
1024/128/128 raw-text-paired dataset from the preceding translator run without copying its 12 GB
cache. Token index 0 is excluded from Writer training. Every receiver call uses Qwen-native token0,
32 translated/native content anchors, canonical positions 0..32, and a suffix beginning at 33.

Phase 1 trains four bias-free, K/V-independent architectures for 512 optimizer steps:

1. `local5_samehead`
2. `full28_samehead`
3. `full28_diagonal`
4. `full28_head8`

Checkpoints at steps 128/256/384/512 are functionally evaluated on validation. Both minimum
choice-KL and maximum-accuracy checkpoints are selected using validation only and reported on test.
The audit directory records 36x28 depth-block norms and, for Head8, 36x8x8 head-block norms.

Run tests with `python tests.py`; launch the study with `bash launch.sh`.
