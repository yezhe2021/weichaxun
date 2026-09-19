# Synchronized-Boundary Native-KV Oracle

Training-free 128-example Oracle audit of cross-tokenizer causal-state pairing. Llama and Qwen token
end boundaries are converted to UTF-8 byte offsets; consecutive shared end boundaries define minimal
synchronized units. Each unit is represented by the Native KV of the rightmost token ending at the
shared boundary—never by pooling or a slot.

The experiment compares Qwen Full, the existing character-point RawAnchor, same-selection
SyncTrigger, Llama SyncMax/SyncSum with duplicate and unique-fill policies, and Qwen-self SyncMax
unique-fill. Every sparse condition uses Qwen-native token0 plus 32 Qwen Native right-boundary KVs at
canonical positions 1..32; the question begins at position 33.

Outputs include accuracy, paired correct/wrong transitions, per-sample unit manifests, and alignment
audits covering unit shapes, duplicates, unique count, and Llama/Qwen selected attention mass.
If a sample contains fewer than 32 synchronized units, unique-fill exhausts every unit and then
deterministically pads with region-best repeats; the unavoidable shortfall is explicitly audited.
