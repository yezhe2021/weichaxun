import torch

from p3e_k_common import InformationSufficiencyProbe


def payload(tokens=9):
    return {
        "canonical_keys": torch.randn(16, tokens, 16, 128),
        "canonical_values": torch.randn(16, tokens, 16, 128),
        "native_keys": torch.randn(16, tokens, 8, 128),
        "native_values": torch.randn(16, tokens, 8, 128),
        "full_text_hidden": torch.randn(tokens, 2560),
        "question": torch.randn(2560),
        "mask": torch.ones(tokens, dtype=torch.bool),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for mode in ("text", "native", "canonical", "zero"):
        model = InformationSufficiencyProbe(mode, max_tokens=16).to(device)
        result = model(payload(), device)
        if result["support"].shape != (9,) or result["yesno"].shape != (2,):
            raise RuntimeError(f"Output shape mismatch for {mode}")
        sum(value.float().mean() for key, value in result.items() if key != "mask").backward()
        if not all(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError(f"Missing gradient in {mode}")
    print("p3e_k_architecture_smoke_ok")


if __name__ == "__main__":
    main()
