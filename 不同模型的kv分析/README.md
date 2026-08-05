# 不同模型的kv分析

本项目用于比较不同模型（Qwen3 全系 / Llama / gemma 等）的 KV cache 行为与分析。

## 目录结构

```
不同模型的kv分析/
├── README.md        # 本文件
├── .gitignore       # 上传规则：脚本+结果上传，参数不上传
├── scripts/         # 实验脚本
└── results/         # 实验结果（指标、日志、图、报告）
```

## 实验环境

- **服务器实验根目录**：`/home/yezhe/不同模型的kv分析`（所有实验在此进行）
- **模型路径**：`/home/yezhe/all_models/models`
- **数据集路径**：`/home/yezhe/数据集`
- **Python 环境**：conda `attnkv`，解释器 `/home/yezhe/data/miniconda3/envs/attnkv/bin/python -u`
- **GPU**：Tesla V100-32GB（CUDA 偶发断连，实验需支持断点续跑）

## 上传策略

- **上传**：实验脚本、实验结果（指标 JSON/CSV、日志、图表、报告）
- **不上传**：实验中产生的模型参数、权重、checkpoint、中间状态（见 `.gitignore`）

工作流：在服务器 `/home/yezhe/不同模型的kv分析` 下跑实验，将脚本与结果同步回本目录，再推送到 GitHub。
