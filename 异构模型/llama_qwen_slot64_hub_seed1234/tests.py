import io

import torch
from torch import nn

from modules import SlotCompressor, RegistrationMapper, full_kl


def run_tests():
    torch.manual_seed(1234)
    compressor = SlotCompressor(5, 4, 2, 4)
    k, v = torch.randn(5, 7, 2, 4), torch.randn(5, 7, 2, 4)
    pk, pv = compressor(k, v)
    assert pk.shape == (5, 4, 2, 4) and pv.shape == pk.shape
    (pk.square().mean() + pv.square().mean()).backward()
    assert compressor.queries.grad is not None and compressor.queries.grad.norm() > 0
    zk, zv = compressor(torch.zeros_like(k), torch.zeros_like(v))
    assert torch.count_nonzero(zk) == 0 and torch.count_nonzero(zv) == 0
    permutation = torch.randperm(7)
    permuted = compressor(k[:, permutation], v[:, permutation])
    torch.testing.assert_close(pk, permuted[0]); torch.testing.assert_close(pv, permuted[1])
    # Adding only V must leave pooled K unchanged: pooling scores depend on K and queries.
    changed = compressor(k, v + 3)
    torch.testing.assert_close(pk, changed[0]); torch.testing.assert_close(pv + 3, changed[1])

    mapper = RegistrationMapper(source=5, target=7, heads=2, dim=4)
    mk, mv = mapper(pk.detach(), pv.detach())
    assert mk.shape == (7, 4, 2, 4)
    mk.sum().backward()
    assert all(p.grad is None for p in mapper.v_depth.parameters())
    assert all(p.grad is None for p in mapper.v_heads.parameters())
    assert mapper.k_depth[0].weight.data_ptr() != mapper.k_depth[1].weight.data_ptr()
    assert all(m.bias is None for m in mapper.modules() if isinstance(m, nn.Linear))
    # Change all trainable weights, then verify exact zero control and slot independence.
    with torch.no_grad():
        for p in mapper.parameters(): p.add_(torch.randn_like(p) * .01)
    z = mapper(torch.zeros_like(pk), torch.zeros_like(pv))
    assert all(torch.count_nonzero(x) == 0 for x in z)
    out = mapper(pk.detach(), pv.detach())
    single = mapper(pk.detach()[:, 1:2], pv.detach()[:, 1:2])
    torch.testing.assert_close(single[0], out[0][:, 1:2]); torch.testing.assert_close(single[1], out[1][:, 1:2])
    stream = io.BytesIO(); torch.save(mapper.state_dict(), stream); stream.seek(0)
    reloaded = RegistrationMapper(5, 7, 2, 4)
    reloaded.load_state_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(reloaded(pk.detach(), pv.detach())[0], out[0], rtol=0, atol=0)
    logits = torch.randn(32, requires_grad=True); teacher = torch.randn(32, requires_grad=True)
    loss = full_kl(logits, teacher); loss.backward()
    assert logits.grad is not None and teacher.grad is None
    torch.testing.assert_close(full_kl(teacher, teacher), torch.tensor(0.), atol=1e-6, rtol=0)
    return {'passed': True, 'checks': ['shapes', 'query_gradients', 'Writer(0)=0', 'bias_free',
            'K/V_independence', 'target_layer_independence', 'slot_independence', 'pooling_weights_shared_KV',
            'pre_RoPE_token_permutation_invariance', 'checkpoint_roundtrip', 'teacher_detached']}


if __name__ == '__main__': print(run_tests())
