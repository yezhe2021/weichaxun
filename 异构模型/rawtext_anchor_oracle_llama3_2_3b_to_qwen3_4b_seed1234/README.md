# Raw-text Anchor Oracle: Llama-3.2-3B to Qwen3-4B

Evaluation-only oracle for a tokenizer-independent 33-entry Native KV interface. Llama Query attention selects one shared character anchor in each of 32 equal raw-text regions; each tokenizer maps that anchor to its own Native token. Position zero is a separate model-native bootstrap channel and is excluded from anchor selection.

The experiment reuses existing Native KV and Query-importance caches and trains no parameters. It separates the cost of raw-character regions from the cost of Llama-driven cross-tokenizer anchors.

```bash
/home/yezhe/data/miniconda3/envs/attnkv/bin/python tests.py
/home/yezhe/data/miniconda3/envs/attnkv/bin/python -u run_pipeline.py --mode smoke
bash launch.sh
tail -f logs/pipeline.log
```
