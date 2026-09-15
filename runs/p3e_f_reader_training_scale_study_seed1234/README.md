# P3-E-F Reader Training Scale Study

This experiment changes only the amount of Reader training data.

## Fixed components

- Qwen3-8B Sender and the P3-E-C2 Writer are frozen.
- Qwen3-4B Receiver backbone is frozen.
- Canonical memory remains `[16, T, 16, 128]`.
- Reader architecture, loss, optimizer, seed, epoch count, decoding, and the
  independent 64-example validation split are fixed.
- `train1024` and `train2048` both load the same P3-E-C1 Reader checkpoint.
  The 2048 run never resumes from the 1024 run.

## Data protocol

- The historical 512 rows and their hard-negative mapping are preserved exactly.
- The 1024 dataset contains the historical 512 prefix.
- The 2048 dataset contains the complete 1024 prefix.
- Validation IDs are excluded.
- New examples whose evidence exceeds 1024 Qwen3-8B tokens are excluded.
- C2 Canonical K/V is cached before Reader training. Reader training never loads
  the Sender or Writer.

## Baseline handling

The 512 model and its fixed-validation outputs are imported from the completed
P3-E-C2 evaluation. They are not retrained.

## Evaluation

Each new Reader is evaluated once after training on:

- `question_only`
- `correct_canonical`
- `hard_shuffled_canonical`
- `oracle_support_canonical`
- `reader_off`

Automatic HotpotQA EM/F1, bridge/comparison splits, prediction switch, EOS,
gates, and supporting-fact attention mass are reported. A blinded C/P/W review
sheet is generated; manual semantic metrics remain explicitly pending until it
is labeled.

## Run and monitor

```bash
bash /home/yezhe/伪查询/runs/p3e_f_reader_training_scale_study_seed1234/launch.sh
tail -f /home/yezhe/伪查询/runs/p3e_f_reader_training_scale_study_seed1234/p3e_f_run.log
```
