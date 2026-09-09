# No-Alignment Pure Functional Translator

Strict NA-1 experiment: randomly initialized Full28 block-diagonal-head Writer trained only through
final-position A-J Choice-KL from a full-context Qwen3-4B teacher. There is no Stage A, content
target-KV reconstruction, gold-label loss, or cross-tokenizer token correspondence in the training
interface.

The Llama-selected content memories are the same 32 query-aware source tokens used in the aligned
experiment and remain in source-text order. The receiver cache is Qwen-native token0 followed by 32
functional memories at canonical positions 1..32; the Qwen query suffix starts at position 33.

Run `python tests.py`, then launch with `bash launch.sh`. Results are written under `runs/study`.
