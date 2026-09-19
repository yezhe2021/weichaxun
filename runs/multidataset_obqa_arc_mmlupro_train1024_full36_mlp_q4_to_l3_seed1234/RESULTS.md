# Qwen3-4B → Llama3.2-3B：三数据集混合训练

正式反向实验完成（seed 1234）。OpenBookQA、ARC-Challenge、MMLU-Pro 各取 1024/128/128 条训练/验证/测试样本；Options KV 预算分别为 32/32/64。标准 Question→Options→Answer 顺序，Receiver 保留原生 Question。Stage-A 训练 1536 steps，Stage-B 冻结 Stage-A 并训练 Residual64，768 steps。

| 测试集（各 128 条） | Stage-A | Residual Stage-B |
| --- | ---: | ---: |
| OpenBookQA | 60.16% | 64.06% |
| ARC-Challenge | 62.50% | 62.50% |
| MMLU-Pro | 25.78% | 30.47% |
| 合计（384 条） | 49.48% | 52.34% |

详细结果见 `runs/study/results/comparison.json`、`per_sample_metrics.jsonl` 和 native baseline 文件。训练曲线、协议审计及完整日志亦保存在本目录。`.pt` checkpoint 与 KV 缓存仅保存在实验服务器，不上传 GitHub。MMLU-Pro 使用 official test 文件的互斥派生划分，因此上表不是官方 benchmark test 分数。
