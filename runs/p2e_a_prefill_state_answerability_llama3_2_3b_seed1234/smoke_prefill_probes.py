import torch
import torch.nn.functional as F

from prefill_probes import build_probe


def main():
    hidden = 64
    classes = 7
    configs_and_shapes = (
        ({"kind": "end_linear"}, (4, hidden), None),
        ({"kind": "summary_linear"}, (4, 8, hidden), torch.ones(4, 8, dtype=torch.bool)),
        ({"kind": "summary_attention"}, (4, 16, hidden), torch.ones(4, 16, dtype=torch.bool)),
        ({"kind": "raw_evidence_attention"}, (4, 23, hidden), torch.ones(4, 23, dtype=torch.bool)),
    )
    for config, shape, mask in configs_and_shapes:
        model = build_probe(config, hidden, classes, attention_rank=16, value_rank=32)
        logits = model(torch.randn(*shape), mask)
        if logits.shape != (4, classes):
            raise RuntimeError(f"Unexpected logits shape for {config}: {tuple(logits.shape)}")
        F.cross_entropy(logits, torch.arange(4) % classes).backward()
        if not any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError(f"No gradients for {config}")
    print("Experiment A synthetic probe smoke passed")


if __name__ == "__main__":
    main()
