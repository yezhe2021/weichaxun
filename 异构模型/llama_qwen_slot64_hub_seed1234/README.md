# Llama-3.2-3B → 64-slot Qwen Hub → Qwen3-4B

Independent pilot under `/home/yezhe/异构模型/llama_qwen_slot64_hub_seed1234`.
Models are loaded locally and frozen; no LoRA or external checkpoint is used.

## Protocol

MMLU-Pro original option text is the sender prefix. Question and answer instruction arrive
only at the receiver. No gold labels, question text or chain-of-thought is inserted in the prefix.
Family-specific tokenizers process the same text separately. BOS is inserted once if defined.
Prefix/suffix compositionality and actual space-letter continuation token IDs are verified.
Predictions are the highest final-position logit among valid options, not open-ended generation.

The default pilot uses 128 train, 32 validation and 128 test examples. They are disjoint,
category-balanced internal subsets of the official MMLU-Pro test pool; these are feasibility
results, not scores on an untouched official benchmark. Filter long samples using both
tokenizers (512 prefix / 384 suffix maximum); keep whole text and original option order.
Smoke uses 4/2/2 and separate outputs under `runs/smoke`.

Capture Qwen K after k_norm, Llama K after k_proj, both before RoPE; V after v_proj.
Each compressor has independent queries `[layers,8,64,128]`. For each layer/head:

`A = softmax(Q @ (K / RMS(K)).T / sqrt(128))`; `K_slot=A@K`, `V_slot=A@V`.

RMS is used only for scoring, over native prefix tokens/features in each layer/head.
Pooling weights are shared between K and V. Only queries are trainable. Every prefix token
participates. T<64 examples are retained and explicitly constitute resampling, not compression.
No token interpolation or matching across model tokenizers is required.

Apply the receiving model's RoPE to slot K at positions 0..63, then start suffix at 64.
Native full-cache suffix starts at its native T. The self-slot diagnostic includes both
the pooling bottleneck and this change of positions. Full-text vs full-cache/manual vs
official cache equality is audited separately for both models before training.

## Training sequence

1. Qwen self-slot: final-position full-vocab KL(Qwen Native || Qwen Slot), queries only.
2. Llama self-slot: final-position full-vocab KL(Llama Native || Llama Slot), queries only.
3. Save the selected self-slot checkpoints independently. Materialize their slots and
   receiver logits. Qwen compressor now defines the frozen Hub anchor.
4. Stage A initializes a fresh Llama compressor from its self-slot checkpoint. Jointly
   train its queries and the depth/head mapper to match frozen Qwen slots with the sum
   over K/V of mean per-layer NMSE + (1-cosine).
5. Stage B starts from Stage A best. Train only that Llama compressor and mapper with
   final-position full-vocab KL(Qwen Slot || Llama→Hub→Qwen). No cross-vocabulary KL.

Mapper: each target layer has an independent bias-free `28*128→128` projection applied
to each source head. Concatenate its resulting 8 latent heads, then apply a separate
`1024→128` W for each target head (stored as row blocks of `1024→1024`). K/V parameters
are independent throughout; there is no slot mixing. This is a factorized linear mapper,
not an unrestricted `28672→1024` matrix. Depth and head maps both train in Stage A/B.
Initialization selects a nearest-depth block and same-head identity; no older experiment
weights are imported. Slot identities are initially arbitrary and must be registered by
the jointly trained Llama queries and mapper; Stage A diagnostics measure that limitation.

Each stage runs two true epochs (one in smoke), batch=8 by gradient accumulation
(2 in smoke), clip=30, AdamW without weight decay. Log every optimizer step and clip norm.
No accuracy/representation thresholds stop feasibility stages. Finite-loss/gradient and
protocol failures do stop execution. Save initial, every epoch, best including initial,
best_trained excluding initial, and last; explicitly record selected epochs.

## Evaluation

Report Qwen Native/Self-Slot, Llama Native/Self-Slot, Stage A best, Stage B best,
Stage B best_trained/last, Qwen question-only, true-zero slots and different-sample
shuffled Llama Hub slots. Keep the original self-slot checkpoints for self diagnostics.

Accuracy uses gold only at evaluation. Agreement/full KL have explicit references:
self-slot vs same-family Native, cross vs Qwen Slot; also report cross vs Qwen Native.
Report R_Q, R_L, R_cross and accuracy gaps. Undefined ratios return null if denominator=0.
Qwen Slot is a reference target, not a guaranteed upper bound on accuracy. High/low
retention suggests directions but cannot prove a unique causal explanation from this pilot.

Outputs: `runs/{smoke,pilot}/results/summary.json`, `per_sample_predictions.jsonl`,
training step JSONL, epoch summaries, checkpoints, manifests, audits and status.json.
Temporary full KV and slot caches stay on the server for reuse; excluded from Git with
checkpoints. No automatic cache deletion is performed.

## Run

```bash
bash launch.sh smoke
tail -f logs/smoke_pipeline.log
bash launch.sh pilot
tail -f logs/pilot_pipeline.log
cat runs/pilot/status.json
```

Runs take a process lock. Resume skips completed stages after matching config/code hashes.
Failed stages rerun from their start; native cache building resumes sample by sample.
Run smoke and pilot sequentially to avoid GPU contention. No continuous external monitor.
