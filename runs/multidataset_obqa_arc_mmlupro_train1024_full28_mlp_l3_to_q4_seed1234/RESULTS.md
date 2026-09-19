# Llama3.2-3B → Qwen3-4B：三数据集混合训练

正式实验完成（seed 1234）。OpenBookQA、ARC-Challenge、MMLU-Pro 各取 1024/128/128 条训练/验证/测试样本；Options KV 预算分别为 32/32/64。标准 Question→Options→Answer 顺序，Receiver 保留原生 Question。Stage-A 训练 1536 steps，Stage-B 冻结 Stage-A 并训练 Residual64，768 steps。

| 测试集（各 128 条） | Stage-A | Residual Stage-B |
| --- | ---: | ---: |
| OpenBookQA | 52.34% | 61.72% |
| ARC-Challenge | 63.28% | 74.22% |
| MMLU-Pro | 9.38% | 23.44% |
| 合计（384 条） | 41.67% | 53.13% |

详细结果见 `runs/study/results/comparison.json` 和 `per_sample_metrics.jsonl`；训练曲线、协议审计及完整日志亦保存在本目录。`.pt` checkpoint 与 KV 缓存仅保存在实验服务器，不上传 GitHub。MMLU-Pro 的本地数据只有 official test 文件，本实验把它确定性划为互斥的派生训练/验证/测试集，因此上表不是官方 benchmark test 分数。
