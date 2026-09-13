# MMLU-Pro Standard-Prompt Full28-MLP

Controlled migration from the prior options-first protocol to direct MCQ order:

```text
Question:
...

Options:
A. ...
...

Answer:
```

This is the capacity-controlled follow-up to the Full28-Diagonal linear baseline. The routing query is the Sender-native final `Answer:` token;
candidates are body tokens only. Selection remains one maximum-attention token per each of 32
normalized raw-text regions, preserving text order. RawAnchor, 32 memories, native Qwen token0,
canonical positions, Full28 MLP (`3584 -> 1024 -> 128` per target layer and K/V, followed by diagonal head adapters), Stage-A reconstruction, Stage-B Oracle choice-KL,
1024/128/128 IDs, seed and training settings remain fixed. All Native KV caches and checkpoints
are held fixed: the exact audited manifests, selected-KV pairs and Oracle teachers from the linear
baseline are reused read-only, while all MLP checkpoints and evaluations are generated from scratch.

The previous 512-token limit applied only to the options prefix and is not reused as a semantic
filter here: the standard body combines the old options prefix and question suffix. A 2048-token
safety ceiling is used without truncation, and observed maxima for both tokenizers are audited.

Pipeline: manifest → Llama fresh cache → Qwen fresh cache/RawAnchor pairs → Phase-0 audit →
fresh baselines/Oracle teacher → Stage-A → validation-only Stage-A selection → Stage-B → evaluation.

For the formal run, `preflight` stops after the 32-sample Phase-0 audit so it can be inspected
before any training starts. Running `all` afterwards reuses the completed preflight stages.

```bash
cd /home/yezhe/异构模型/mmlupro_standardprompt_rawanchor_full28_mlp_llama3_2_3b_to_qwen3_4b_seed1234
nohup bash launch_when_cuda_ready.sh > logs/launcher.log 2>&1 &
tail -f logs/pipeline.log
```

Final results are written to `runs/study/results/comparison.json` and automatically appended to
`/home/yezhe/异构模型/EXPERIMENT_RESULTS.md`. Do not upload `.pt`, cache, or checkpoints.
