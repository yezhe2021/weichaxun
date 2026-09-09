import argparse
import copy
import random
from pathlib import Path

import torch
from tqdm import tqdm

from p3d3_common import seed_everything, write_json, write_jsonl
from p3e_n2_common import (
    ReadoutCache,
    ReadoutContentProbe,
    payload_to_device,
    probe_loss,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    cache = ReadoutCache(args.cache)
    total = min(args.max_samples, len(cache))
    indices = list(range(total))
    probe = ReadoutContentProbe().to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    history = []
    best_loss = float("inf")
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        order = indices.copy()
        random.Random(args.seed + epoch).shuffle(order)
        probe.train()
        totals = {"loss": [], "support": [], "span": [], "yesno": []}
        for index in tqdm(order, desc=f"p3e_n2_probe_epoch{epoch}"):
            payload = cache.load(index)
            readout, attention, labels = payload_to_device(
                payload, "correct", device
            )
            optimizer.zero_grad(set_to_none=True)
            outputs = probe(
                readout, attention, labels["valid_mask"]
            )
            loss, parts = probe_loss(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()
            totals["loss"].append(float(loss.detach()))
            for name, value in parts.items():
                if value is not None:
                    totals[name].append(float(value.detach()))
        record = {
            "epoch": epoch,
            **{
                f"train_{name}": sum(values) / len(values)
                for name, values in totals.items()
                if values
            },
        }
        history.append(record)
        write_jsonl(output / "training_history.jsonl", history)
        if record["train_loss"] < best_loss:
            best_loss = record["train_loss"]
            best_epoch = epoch
            state = copy.deepcopy(
                {name: tensor.detach().cpu() for name, tensor in probe.state_dict().items()}
            )
            torch.save(
                {
                    "probe": state,
                    "epoch": epoch,
                    "train_loss": best_loss,
                    "architecture": {
                        "readout": "[16,2560] structured projection",
                        "attention": "[16,32,T] no layer/head averaging",
                        "token_mixer": "2-layer bidirectional Transformer",
                        "heads": ["support", "span", "yes_no"],
                    },
                    "reader_receiver_frozen_and_absent_from_optimizer": True,
                },
                output / "checkpoint_best.pt",
            )
    write_json(
        output / "TRAIN_SUCCESS.json",
        {
            "status": "complete",
            "experiment": "P3-E-N2 Reader readout content probe",
            "samples": total,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_train_loss": best_loss,
            "checkpoint": str(output / "checkpoint_best.pt"),
            "only_probe_trained": True,
        },
    )


if __name__ == "__main__":
    main()
