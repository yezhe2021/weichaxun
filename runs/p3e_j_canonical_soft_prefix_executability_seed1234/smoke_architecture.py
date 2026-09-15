import torch

from p3e_j_common import SoftPrefixDecoder


def main():
    torch.manual_seed(1234)
    decoder = SoftPrefixDecoder(max_tokens=16)
    keys = torch.randn(16, 7, 16, 128)
    values = torch.randn_like(keys)
    mask = torch.ones(7, dtype=torch.bool)
    output, diagnostics = decoder(keys, values, mask, return_diagnostics=True)
    if output.shape != (7, 2560):
        raise RuntimeError(f"Unexpected output shape {tuple(output.shape)}")
    output.square().mean().backward()
    if not all(parameter.grad is not None for parameter in decoder.parameters()):
        raise RuntimeError("A Soft Prefix Decoder parameter did not receive gradients")
    if diagnostics["head_weights"].shape != (16, 7, 16):
        raise RuntimeError("Head pooling diagnostic shape mismatch")
    if diagnostics["layer_weights"].shape != (16, 7):
        raise RuntimeError("Layer mixing diagnostic shape mismatch")
    print("soft_prefix_architecture_smoke_ok")


if __name__ == "__main__":
    main()
