import torch

from common import build_anchors, serialize, Span
from translators import DepthHeadTranslator, ResidualAdapter


row = {"id": "x", "question": "Why?", "options": ["one", "two", "three", "four"], "gold_index": 2}
serialized = serialize(row)
assert serialized["labels"] == list("ABCD")
assert serialized["full_prompt"].endswith("Answer:")

spans = [Span(1, serialized["options_char_span"][0], serialized["options_char_span"][1])]
importance = torch.tensor([0.0, 1.0])
anchors = build_anchors(serialized["body"], spans, spans, importance, 32,
                        *serialized["options_char_span"])
assert len(anchors) == 32

small = DepthHeadTranslator(2, 3, 2, 2, 4, 4, 5, 4, "per_head")
k = torch.randn(1, 2, 32, 2, 4); v = torch.randn_like(k)
pk, pv = small(k, v)
assert pk.shape == pv.shape == (1, 3, 32, 2, 4)
adapter = ResidualAdapter(3, 2, 4, 2)
ak, av = adapter(pk, pv)
assert torch.equal(ak, pk) and torch.equal(av, pv)
assert all(module.bias is None for module in small.modules() if isinstance(module, torch.nn.Linear))
assert all(module.bias is None for module in adapter.modules() if isinstance(module, torch.nn.Linear))
print("PASS: datasets, 32-region anchors, translators, residual zero-init, bias-free linears")
