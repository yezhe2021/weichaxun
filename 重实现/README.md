# 重实现

本目录用于存放重新建立实验协议后的独立实验。

服务器对应目录：

```text
/home/yezhe/重实现
```

## 默认实验约束

- Sender 处理完整 Context/Evidence，不提前读取 Question。
- 传输完整 Context 的全 token Native KV，不使用 supporting-token、supporting-fact 或其他 sparse oracle KV。
- Writer 的每个目标层使用独立参数，不跨层共享 Linear 或 MLP。
- K Writer 与 V Writer 分开建模、分开映射。
- 默认使用 `bias=False`，并测试 `Writer(0)=0`。
- Correct、Shuffled、Zero 与 Native controls 使用相同的 token、position IDs、mask 和注入协议。
- 实验以快速机制验证为目标；先运行小规模 smoke/diagnostic，再决定是否扩大。

## 上传规则

每个实验使用独立子目录。上传代码、配置、轻量 JSON/CSV 结果和必要日志；不上传模型、checkpoint、KV cache、数据集副本或其他大型中间文件。
