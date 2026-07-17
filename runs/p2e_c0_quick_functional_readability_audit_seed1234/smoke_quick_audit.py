import torch
import torch.nn.functional as F

from functional_probes import HiddenTokenProbe, KVFunctionalProbe, LayerStateProbe, VectorStateProbe


def check(model, feature, classes):
    logits = model(feature)
    if logits.shape != (classes,):
        raise RuntimeError(f"Unexpected shape: {tuple(logits.shape)}")
    F.cross_entropy(logits.unsqueeze(0), torch.tensor([1])).backward()
    if not any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Probe did not receive gradients")


def main():
    classes = 40
    check(HiddenTokenProbe(96, 32, classes), torch.randn(17, 96), classes)
    memory = {
        "keys": [torch.randn(4, 19, 16) for _ in range(3)],
        "values": [torch.randn(4, 19, 16) for _ in range(3)],
    }
    check(KVFunctionalProbe(3, 16, 32, classes), memory, classes)
    check(LayerStateProbe(128, 32, classes), torch.randn(5, 128), classes)
    check(VectorStateProbe(128, 32, classes), torch.randn(128), classes)
    print("P2-E-C0 synthetic probe smoke passed")


if __name__ == "__main__":
    main()
