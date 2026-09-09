import torch

from modules import HybridChunkCompressor, full_kl


def run_tests():
    torch.manual_seed(1234)
    module = HybridChunkCompressor(2, slots=4, heads=2, dim=4)
    key, value = torch.randn(2, 13, 2, 4), torch.randn(2, 13, 2, 4)
    importance = torch.tensor([0., 2., 1., 0., 1., 5., 0., 1., 0., 0., 3., 1., 0.])
    query = module.select(importance, "query", 9)
    center = module.select(importance, "center", 9)
    random1 = module.select(importance, "random", 9)
    random2 = module.select(importance, "random", 9)
    assert query.tolist() == [1, 5, 7, 10]
    assert center.tolist() == [1, 4, 7, 10]
    assert torch.equal(random1, random2)
    weights, native_valid, slot_valid = module.attention(key, query)
    assert native_valid.all() and slot_valid.all()
    for i, index in enumerate(query): assert torch.count_nonzero(weights[:, :, i, index]) == 0
    outputs = module(key, value, query)
    joint_k, joint_v, valid = module.interleave(key[:, :1], value[:, :1], outputs)
    assert joint_k.shape == joint_v.shape == (2, 9, 2, 4) and valid.shape == (9,)
    for i, index in enumerate(query):
        torch.testing.assert_close(joint_k[:, 1 + 2 * i], key[:, index])
        torch.testing.assert_close(joint_v[:, 1 + 2 * i], value[:, index])
    short = HybridChunkCompressor(2, slots=8, heads=2, dim=4)
    sk, sv = torch.randn(2, 3, 2, 4), torch.randn(2, 3, 2, 4)
    selected = short.select(torch.ones(3), "center", 1)
    result = short(sk, sv, selected)
    assert result[4].sum() == 3 and result[5].sum() == 0
    memory = short.interleave(sk[:, :1], sv[:, :1], result)
    assert memory[0].shape[1] == 17 and memory[2].sum() == 4
    student, teacher = torch.randn(20, requires_grad=True), torch.randn(20, requires_grad=True)
    full_kl(student, teacher).backward()
    assert student.grad is not None and teacher.grad is None
    return {"passed": True, "checks": ["balanced_chunks", "unified_token_index", "query_top1",
            "center", "deterministic_random", "selected_excluded_from_slot", "interleaved_ANS_schema",
            "short_chunk_masks", "teacher_detached"]}


if __name__ == "__main__": print(run_tests())
