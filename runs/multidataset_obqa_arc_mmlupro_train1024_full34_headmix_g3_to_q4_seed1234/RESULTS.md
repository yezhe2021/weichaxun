# Gemma3-4B → Qwen3-4B：三数据集混合训练

正式实验完成（seed 1234）。OpenBookQA、ARC-Challenge、MMLU-Pro 各取 1024/128/128 条训练/验证/测试样本；Options KV 预算分别为 32/32/64。标准 Question→Options→Answer 顺序，Receiver 保留原生 Question。Full34 HeadMix256 Stage-A 训练 1536 steps，冻结基座 Residual64 Stage-B 训练 768 steps。

| 测试集（各 128 条） | Stage-A | Residual Stage-B |
| --- | ---: | ---: |
| OpenBookQA | 41.41% | 70.31% |
| ARC-Challenge | 56.25% | 74.22% |
| MMLU-Pro | 10.94% | 30.47% |
| 合计（384 条） | 36.20% | 58.33% |

Receiver native Oracle（按相同选点）合计 68.49%。详细结果、native controls、训练曲线和逐样本指标见 `runs/study/`。某些机器可读的 baseline 键名沿用 `selected32/oracle32`，但本次 MMLU-Pro 实际使用 64 个 Options KV，以 `config.json`、协议字符串及逐样本缓存记录为准。`.pt` checkpoint 与 KV 缓存仅保存在实验服务器。MMLU-Pro 分数基于 official test 文件的互斥派生划分，不是官方 benchmark test 分数。
