# Posterior-Question Sender Routing

This controlled experiment changes only the Sender routing sequence relative to the standard-prompt
native-question baseline:

```text
Sender:   Question -> Options -> repeated Question
          final repeated-Question token routes over Options only

Receiver: native Question -> 32 translated Options KVs -> native Answer:
```

The 32-region coverage constraint, RawAnchor, Full28-Diagonal bias-free K/V-independent Writer,
1024/128/128 MMLU-Pro IDs, seed 1234, Stage-A reconstruction, 512-step Stage-B choice-KL, and
Q-first Receiver are unchanged. The original standard-prompt cache supplies only unchanged native
accuracy controls; routing selections and source/target KV pairs are freshly generated.

The experiment tests whether the old Options-first protocol benefited because Question appeared
after the candidates and therefore supplied a stronger routing state. It does not mix in the
separately tested Memory-first Receiver.

Pipeline: manifest → Llama fresh cache → Qwen fresh cache/RawAnchor pairs → Phase-0 audit →
fresh baselines/Oracle teacher → Stage-A → validation-only Stage-A selection → Stage-B → evaluation.

For the formal run, `preflight` stops after the 32-sample Phase-0 audit so it can be inspected
before any training starts. Running `all` afterwards reuses the completed preflight stages.

```bash
cd /home/yezhe/异构模型/mmlupro_sender_qoq_posterior_question_rawanchor_full28_diagonal_llama3_2_3b_to_qwen3_4b_seed1234
nohup bash launch_when_cuda_ready.sh > logs/launcher.log 2>&1 &
tail -f logs/pipeline.log
```

Final results are written to `runs/study/results/comparison.json` and automatically appended to
`/home/yezhe/异构模型/EXPERIMENT_RESULTS.md`. Do not upload `.pt`, cache, or checkpoints.
