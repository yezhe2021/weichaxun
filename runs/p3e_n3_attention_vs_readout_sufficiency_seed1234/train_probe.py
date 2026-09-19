import argparse
import copy
import random
from pathlib import Path

import torch
from tqdm import tqdm

from p3d3_common import seed_everything, write_json, write_jsonl
from p3e_n2_common import ReadoutCache, payload_to_device, probe_loss
from p3e_n3_common import SufficiencyProbe, VALID_MODES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=VALID_MODES, required=True)
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
    probe = SufficiencyProbe(args.mode).to(device)
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
        for index in tqdm(
            order, desc=f"p3e_n3_{args.mode}_epoch{epoch}"
        ):
            payload = cache.load(index)
            readout, attention, labels = payload_to_device(
                payload, "correct", device
            )
            optimizer.zero_grad(set_to_none=True)
            outputs = probe(readout, attention, labels["valid_mask"])
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
                {
                    name: tensor.detach().cpu()
                    for name, tensor in probe.state_dict().items()
                }
            )
            torch.save(
                {
                    "probe": state,
                    "mode": args.mode,
                    "epoch": epoch,
                    "train_loss": best_loss,
                    "reader_c_frozen_and_not_loaded": True,
                    "native_kv_excluded": True,
                },
                output / "checkpoint_best.pt",
            )

    write_json(
        output / "TRAIN_SUCCESS.json",
        {
            "status": "complete",
            "experiment": "P3-E-N3 Attention-vs-Readout Sufficiency Ablation",
            "mode": args.mode,
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

