# P3-D3 Layer-Aligned Native-Query Reader

This is a small-scale structural ablation on official HotpotQA train/dev data. It tests whether a frozen Qwen3-4B can execute question-independent multi-layer evidence memory through the shortest layer-aligned path.

## Fixed protocol

- Sender: frozen Qwen3-8B, **Evidence only**.
- Memory groups: the P3-C stable `uniform16` layers in fixed normalized-depth order.
- Canonical memory: frozen P3-C Writer output `[16, T, 256]`.
- Receiver: frozen Qwen3-4B.
- Reader layers: the same 16 normalized-depth indices, one-to-one with memory groups.
- Query: exact receiver `q_proj -> reshape -> q_norm` output before RoPE, flattened from `32 x 128` and mapped by a rank-32 adapter.
- Injection: external attention branch is added to native `self_attn` output before the DecoderLayer attention residual and MLP.
- Gate: direct unconstrained scalar initialized to `0.01`.

There is no dynamic layer router, cross-layer fusion, compatibility gate, residual MLP, grounding distillation, execution matching, or `INSUFFICIENT` objective.

## Training

`canonical16` and `native_projected16` use separate, capacity-matched Readers. The latter is a projected Native control, not a strict native GQA upper bound. Each training loss is:

```text
mean answer-token NLL(correct memory)
+ 0.5 * max(0, 0.5 + NLL(correct) - NLL(hard shuffled))
```

Hard negatives match question type and answer type while excluding answer aliases, overlapping supporting titles, repeated bridge entities, and evidence containing the current answer. Checkpoint selection uses only the training objective. The official validation subset is evaluated once after both Readers finish.

## Run

```bash
cd /home/yezhe/伪查询/runs/p3d3_layer_aligned_native_query_reader_seed1234
chmod +x run_all.sh
nohup bash run_all.sh all > p3d3_run.log 2>&1 & echo $! > p3d3_run.pid
```

Stages can be run independently with `prepare`, `cache`, `train-canonical`, `train-native`, and `eval`.

```bash
tail -f /home/yezhe/伪查询/runs/p3d3_layer_aligned_native_query_reader_seed1234/p3d3_run.log
```

The final report is `eval/SUCCESS.json`; all raw generations are in `eval/per_sample_generation.jsonl`.
