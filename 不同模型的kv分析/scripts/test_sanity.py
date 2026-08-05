"""最小单元测试（审查 §五）：正式跑 Development 前的四个门禁。

  test_direction_metadata   方向元数据：Sender/Receiver 模型映射正确（问题 5）
  test_checkpoint_roundtrip checkpoint 保存→加载后输出一致（问题 3）
  test_writer_gradient      Writer 前向保留梯度、backward 后参数收到梯度（问题 1）
  test_self_identity_gpu    真模型上 Identity Writer 与 Native Cache 一致（问题 11，需 GPU+模型）

用法：
  python -u test_sanity.py                        # 仅 CPU 逻辑测试
  python -u test_sanity.py --gpu --model <path>   # 附加真模型 self-identity 测试
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parent


def test_direction_metadata():
    sys.path.insert(0, str(SCRIPTS))
    from run_pipeline import EXPERIMENTS, MODELS
    # 17_to_06：Sender 必须 1.7B，Receiver 必须 0.6B
    dirs = {d[0]: d for d in EXPERIMENTS["06_17"]["directions"]}
    _, _, _, s, r = dirs["17_to_06"]
    assert s == "1.7B", f"17_to_06 sender should be 1.7B, got {s}"
    assert r == "0.6B", f"17_to_06 receiver should be 0.6B, got {r}"
    _, _, _, s2, r2 = dirs["06_to_17"]
    assert s2 == "0.6B" and r2 == "1.7B"
    # self 方向 Sender==Receiver
    _, _, _, s3, r3 = dirs["self_17"]
    assert s3 == r3 == "1.7B"
    dirs4 = {d[0]: d for d in EXPERIMENTS["4b_8b"]["directions"]}
    _, _, _, s4, r4 = dirs4["08_to_04"]
    assert s4 == "8B" and r4 == "4B"
    print("PASS test_direction_metadata")


def test_checkpoint_roundtrip():
    from writer import LinearWriter, load_writer, save_writer
    cfg = {"num_layers": 2, "feature_dim": 1024}
    scales = {
        name: torch.rand(2, 1024) + 0.5
        for name in ("source_k", "source_v", "target_k", "target_v")
    }
    writer = LinearWriter(scales, cfg)
    x = torch.randn(2, 5, 8, 128)
    out1 = writer(x, x)[0]
    path = Path("/tmp/test_writer_roundtrip.pt")
    save_writer(path, writer, cfg, {"probe": 1})
    writer2, metadata = load_writer(path, map_location="cpu")
    out2 = writer2(x, x)[0]
    assert torch.allclose(out1, out2), "checkpoint roundtrip output mismatch"
    assert metadata.get("probe") == 1
    path.unlink(missing_ok=True)
    print("PASS test_checkpoint_roundtrip")


def test_writer_gradient():
    from writer import LinearWriter
    cfg = {"num_layers": 2, "feature_dim": 1024}
    scales = {
        name: torch.rand(2, 1024) + 0.5
        for name in ("source_k", "source_v", "target_k", "target_v")
    }
    writer = LinearWriter(scales, cfg)
    x = torch.randn(2, 5, 8, 128)
    key, value = writer(x, x)
    loss = (key.float().pow(2) + value.float().pow(2)).mean()
    assert loss.requires_grad, "writer output must keep gradient"
    loss.backward()
    assert writer.weight_k.grad is not None and writer.weight_k.grad.abs().sum().item() > 0, "weight_k has no gradient"
    assert writer.weight_v.grad is not None and writer.weight_v.grad.abs().sum().item() > 0, "weight_v has no gradient"
    print("PASS test_writer_gradient")


def test_self_identity_gpu(model_path):
    """真模型：Identity Writer（scale=1，即不做 scale 交换）应原样返回输入 KV（问题 11）。

    Self 方向的真实语义是 source/target 用同一模型同一组 RMS scale，identity 还原为原 KV；
    这里用 scale=1 验证 Identity 线性层本身不引入任何改变。
    """
    from protocol import capture_native, cuda, load_model, render, tokenizer
    cfg = {
        "receiver_dtype": "float16",
        "max_new_tokens": 16,
    }
    model = load_model(model_path, cfg)
    tok = tokenizer(model_path)
    num_layers = model.config.num_hidden_layers
    raw = {"context": [[f"T{i}", [f"sentence {j} " for j in range(3)]] for i in range(3)],
           "question": "What is the answer?",
           "answer": "yes",
           "type": "bridge"}
    sample = render(tok, raw, True)
    kv = capture_native(model, sample["context_input_ids"], num_layers)
    from writer import LinearWriter
    scales = {
        name: torch.ones(num_layers, 1024)
        for name in ("source_k", "source_v", "target_k", "target_v")
    }
    writer = LinearWriter(scales, {"num_layers": num_layers, "feature_dim": 1024}).to(cuda()).eval()
    ik, iv = writer(kv[0].to(cuda()), kv[1].to(cuda()))
    error = max((ik.float() - kv[0].float().to(cuda())).abs().max().item(),
                (iv.float() - kv[1].float().to(cuda())).abs().max().item())
    print(f"identity kv max error: {error:.3e}")
    assert error < 1e-3, f"identity writer should be near-native, max error {error}"
    print("PASS test_self_identity_gpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--model")
    args = parser.parse_args()
    test_direction_metadata()
    test_checkpoint_roundtrip()
    test_writer_gradient()
    if args.gpu:
        if not args.model:
            raise ValueError("--gpu requires --model")
        test_self_identity_gpu(args.model)
    print("ALL SANITY TESTS PASSED")


if __name__ == "__main__":
    main()
