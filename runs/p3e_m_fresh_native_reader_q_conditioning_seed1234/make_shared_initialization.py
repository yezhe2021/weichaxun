import argparse
from pathlib import Path

import torch

from p3d3_common import SELECTED_LAYERS, load_receiver, seed_everything, write_json
from p3e_a_common import NativeHeadwiseReader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--gate-init", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    seed_everything(args.seed)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    model, _ = load_receiver(args.model, torch.device(args.device))
    reader = NativeHeadwiseReader(
        model, SELECTED_LAYERS, rank=args.rank, gate_init=args.gate_init
    ).cpu()
    state = {name: tensor.detach().clone() for name, tensor in reader.state_dict().items()}
    torch.save(
        {
            "reader": state,
            "reader_metadata": reader.metadata(),
            "seed": args.seed,
            "purpose": "identical fresh initialization for Reader A and Reader B",
            "loads_existing_reader_weights": False,
        },
        output,
    )
    write_json(
        output.with_suffix(".json"),
        {
            "status": "complete",
            "checkpoint": str(output),
            "seed": args.seed,
            "reader_metadata": reader.metadata(),
            "loads_existing_reader_weights": False,
        },
    )


if __name__ == "__main__":
    main()
