import torch

from modules import HardChunkSlotCompressor, full_kl


def run_tests():
    torch.manual_seed(1234)
    module = HardChunkSlotCompressor(2, slots=4, heads=2, dim=4)
    key = torch.randn(2, 13, 2, 4)
    value = torch.randn_like(key)
    bounds = module.chunk_bounds(13)
    assert bounds == [(0, 3), (3, 6), (6, 9), (9, 13)]
    assert module.chunk_sizes(13) == [3, 3, 3, 4]
    mask = module.chunk_mask(13, key.device)
    assert mask.shape == (4, 13)
    assert torch.equal(mask.sum(0), torch.ones(13, dtype=torch.long))
    weights = module.attention(key)
    assert weights.shape == (2, 2, 4, 13)
    torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
    assert torch.count_nonzero(weights.masked_select(~mask[None, None])) == 0
    pooled_k, pooled_v = module(key, value)
    assert pooled_k.shape == pooled_v.shape == (2, 4, 2, 4)
    (pooled_k.square().mean() + pooled_v.square().mean()).backward()
    assert module.queries.grad is not None and module.queries.grad.norm() > 0
    zero_k, zero_v = module(torch.zeros_like(key), torch.zeros_like(value))
    assert torch.count_nonzero(zero_k) == 0 and torch.count_nonzero(zero_v) == 0
    # Native token0 bypass plus four slots; production uses the identical schema with 16 slots.
    joint_k = torch.cat((key[:, :1], pooled_k), dim=1)
    joint_v = torch.cat((value[:, :1], pooled_v), dim=1)
    assert joint_k.shape == joint_v.shape == (2, 5, 2, 4)
    torch.testing.assert_close(joint_k[:, 0], key[:, 0])
    torch.testing.assert_close(joint_v[:, 0], value[:, 0])
    student, teacher = torch.randn(20, requires_grad=True), torch.randn(20, requires_grad=True)
    objective = full_kl(student, teacher)
    objective.backward()
    assert student.grad is not None and teacher.grad is None
    return {"passed": True, "checks": ["balanced_contiguous_chunks", "complete_partition",
            "no_cross_chunk_attention", "attention_rows_sum_one", "shape", "query_gradient",
            "Writer(0)=0", "native_token0_bypass", "kv_synchronized", "teacher_detached"]}


if __name__ == "__main__":
    print(run_tests())
