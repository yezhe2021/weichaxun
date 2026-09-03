import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    args = parser.parse_args()
    payload = torch.load(Path(args.sample), map_location="cpu", weights_only=False)
    print("keys:", tuple(payload))
    for name, value in payload.items():
        if torch.is_tensor(value):
            print(name, tuple(value.shape), value.dtype)
        elif isinstance(value, dict):
            print(name, "dict", tuple(value))
            for child_name, child_value in value.items():
                if torch.is_tensor(child_value):
                    print(" ", child_name, tuple(child_value.shape), child_value.dtype)
                elif isinstance(child_value, (str, int, float, bool, list, tuple)):
                    rendered = repr(child_value)
                    print(" ", child_name, rendered[:500])
        else:
            print(name, type(value).__name__, repr(value)[:500])


if __name__ == "__main__":
    main()
