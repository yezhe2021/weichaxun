# Raw-text Anchor Native-KV Translator: Llama-3.2-3B → Qwen3-4B

Slot-free, unidirectional 33-entry pre-RoPE KV translation using Llama-driven raw-text anchors. The target is always Qwen Native KV at the token covering the same Llama-selected character anchor.

First-round arms:

- E0: Qwen anchor Oracle.
- E1: normalized-depth nearest-layer direct copy, with no learned value transform.
- E2: nearest-layer bias-free Linear Stage-A.
- E3: Local5 bias-free Linear Stage-A.
- E4: E3 followed by final-position choice-only KL Stage-B.

Every target layer owns separate K and V maps. Maps are shared across the eight corresponding KV heads and never mix heads or entries. Stage-A checkpoints and Stage-B checkpoints are selected by validation choice KL; gold accuracy is recorded but never used for training or checkpoint selection. E3/E4 include the Native-token0/translated-anchor four-way diagnostic.

```bash
/home/yezhe/data/miniconda3/envs/attnkv/bin/python tests.py
/home/yezhe/data/miniconda3/envs/attnkv/bin/python -u run_pipeline.py --mode smoke
bash launch.sh
tail -f logs/pipeline.log
```
