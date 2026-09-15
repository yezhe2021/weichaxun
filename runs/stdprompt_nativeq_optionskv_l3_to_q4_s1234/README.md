# MMLU-Pro Standard-Prompt Native-Question Options-KV

Query-native communication under direct MCQ order:

```text
Question:
...

Options:
A. ...
...

Answer:
```

The Sender sees the full standard prompt and uses its final native `Answer:` token as the routing
query, but candidates are restricted to Options. One token is selected from each of 32 normalized
option-text regions. The Receiver natively prefills Question plus the `Options:` header, consumes
32 translated option KVs, then natively processes the separator and `Answer:`. The Oracle teacher
uses the identical Receiver-native Question and Answer states, replacing only the translated option
KVs with RawAnchor-aligned native Qwen option KVs. RawAnchor, Full28-Diagonal, Stage-A reconstruction,
Stage-B Oracle choice-KL, 1024/128/128 IDs, seed and training settings remain fixed.

The previous 512-token limit applied only to the options prefix and is not reused as a semantic
filter here: the standard body combines the old options prefix and question suffix. A 2048-token
safety ceiling is used without truncation, and observed maxima for both tokenizers are audited.

Pipeline: manifest → Llama fresh cache → Qwen fresh cache/RawAnchor pairs → Phase-0 audit →
fresh baselines/Oracle teacher → Stage-A → validation-only Stage-A selection → Stage-B → evaluation.

For the formal run, `preflight` stops after the 32-sample Phase-0 audit so it can be inspected
before any training starts. Running `all` afterwards reuses the completed preflight stages.

```bash
cd /home/yezhe/异构模型/mmlupro_standardprompt_nativequestion_optionskv_rawanchor_full28_diagonal_llama3_2_3b_to_qwen3_4b_seed1234
nohup bash launch_when_cuda_ready.sh > logs/launcher.log 2>&1 &
tail -f logs/pipeline.log
```

Final results are written to `runs/study/results/comparison.json` and automatically appended to
`/home/yezhe/异构模型/EXPERIMENT_RESULTS.md`. Do not upload `.pt`, cache, or checkpoints.
