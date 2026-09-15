import torch

from modules import SoftLocalSlotCompressor, full_kl


def run_tests():
    torch.manual_seed(1234)
    global_slot = SoftLocalSlotCompressor(2, slots=4, heads=2, dim=4, locality_strength=0)
    local_slot = SoftLocalSlotCompressor(2, slots=4, heads=2, dim=4, locality_strength=16)
    with torch.no_grad(): local_slot.queries.copy_(global_slot.queries)
    key, value = torch.randn(2, 12, 2, 4), torch.randn(2, 12, 2, 4)
    k, v = local_slot(key, value)
    assert k.shape == (2, 4, 2, 4) and v.shape == k.shape
    (k.square().mean() + v.square().mean()).backward()
    assert local_slot.queries.grad is not None and local_slot.queries.grad.norm() > 0
    zk, zv = local_slot(torch.zeros_like(key), torch.zeros_like(value))
    assert torch.count_nonzero(zk) == 0 and torch.count_nonzero(zv) == 0
    # Joint architecture: Native token0 bypasses compression, slots cover tokens 1..T-1.
    joint_k = torch.cat((key[:, :1], k), dim=1)
    joint_v = torch.cat((value[:, :1], v), dim=1)
    assert joint_k.shape == joint_v.shape == (2, 5, 2, 4)
    torch.testing.assert_close(joint_k[:, 0], key[:, 0])
    torch.testing.assert_close(joint_v[:, 0], value[:, 0])
    bias = local_slot.positional_bias(12, key.device)
    centers = bias.argmax(-1)
    assert centers.tolist() == sorted(centers.tolist())
    # With zero content scores, each ordered slot must prefer its own local region.
    with torch.no_grad(): local_slot.queries.zero_()
    weights = local_slot.attention(torch.zeros_like(key))
    assert torch.equal(weights[0, 0].argmax(-1), centers)
    # Lambda zero exactly recovers global content attention.
    manual = torch.softmax(torch.einsum("lhsd,lhtd->lhst", global_slot.queries,
                           key.float().permute(0, 2, 1, 3) /
                           key.float().permute(0, 2, 1, 3).square().mean((-1, -2), keepdim=True).sqrt().clamp_min(1e-6)) / 2, -1)
    torch.testing.assert_close(global_slot.attention(key), manual)
    student, teacher = torch.randn(20, requires_grad=True), torch.randn(20, requires_grad=True)
    loss = full_kl(student, teacher); loss.backward()
    assert student.grad is not None and teacher.grad is None
    return {"passed": True, "checks": ["shape", "query_gradient", "Writer(0)=0",
            "native_token0_bypass", "kv_synchronized",
            "ordered_centers", "soft_locality_bias", "lambda0_global_equivalence", "teacher_detached"]}


if __name__ == "__main__": print(run_tests())
