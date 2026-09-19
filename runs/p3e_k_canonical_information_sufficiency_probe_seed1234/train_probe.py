import argparse
import random
from pathlib import Path

import torch
from tqdm import tqdm

from p3e_k_common import (
    InformationSufficiencyProbe,
    ProbeCache,
    probe_loss,
    write_json,
    write_jsonl,
)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint(model, args, epoch, history):
    return {
        "probe": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "probe_metadata": model.metadata(),
        "mode": args.mode,
        "epoch": int(epoch),
        "history": history,
        "args": vars(args),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text", "native", "canonical", "zero"], required=True)
    parser.add_argument("--canonical-index", required=True)
    parser.add_argument("--native-index", required=True)
    parser.add_argument("--sidecar-index", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cache = ProbeCache(
        args.canonical_index, args.native_index,
        args.sidecar_index, args.data, capacity=2,
    )
    count = min(args.max_samples, len(cache))
    if args.max_samples == 512 and count != 512:
        raise RuntimeError("Formal P3-E-K training requires 512 samples")
    probe = InformationSufficiencyProbe(args.mode).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    expected = {id(parameter) for parameter in probe.parameters()}
    actual = {
        id(parameter)
        for group in optimizer.param_groups for parameter in group["params"]
    }
    if expected != actual:
        raise RuntimeError("Optimizer must contain exactly Probe parameters")
    history, best_loss, best_epoch = [], float("inf"), -1
    indices = list(range(count))
    for epoch in range(1, args.epochs + 1):
        order = indices.copy()
        random.Random(args.seed + epoch).shuffle(order)
        totals = {
            "loss": [], "support_loss": [], "span_loss": [],
            "yesno_loss": [], "gradient_norm": [],
        }
        for index in tqdm(order, desc=f"p3e_k_{args.mode}_epoch{epoch}"):
            payload = cache.load(index)
            optimizer.zero_grad(set_to_none=True)
            result = probe(payload, device)
            loss, components = probe_loss(result, payload)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite Probe loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                probe.parameters(), args.grad_clip,
            )
            if not torch.isfinite(gradient):
                raise RuntimeError("Non-finite Probe gradient")
            optimizer.step()
            totals["loss"].append(float(loss.detach()))
            totals["support_loss"].append(float(components["support_loss"].detach()))
            totals["span_loss"].append(float(components["span_loss"].detach()))
            totals["yesno_loss"].append(float(components["yesno_loss"].detach()))
            totals["gradient_norm"].append(float(gradient))
        summary = {
            "epoch": epoch,
            **{name: sum(values) / len(values) for name, values in totals.items()},
        }
        history.append(summary)
        if summary["loss"] < best_loss:
            best_loss, best_epoch = summary["loss"], epoch
            torch.save(
                checkpoint(probe, args, epoch, history),
                output / "checkpoint_best.pt",
            )
        torch.save(
            checkpoint(probe, args, epoch, history),
            output / "checkpoint_last.pt",
        )
        write_jsonl(output / "training_history.jsonl", history)
    write_json(output / "TRAIN_SUCCESS.json", {
        "status": "complete",
        "experiment": "P3-E-K Canonical Information Sufficiency Probe",
        "mode": args.mode,
        "samples": count,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_training_loss": best_loss,
        "history": history,
        "probe_metadata": probe.metadata(),
        "trainable_parameters": sum(parameter.numel() for parameter in probe.parameters()),
        "receiver_parameters_loaded": 0,
        "sender_parameters_loaded": 0,
        "writer_parameters_loaded": 0,
        "cache_tensors_require_grad": False,
        "optimizer_boundary": "probe_parameters_only",
    })


if __name__ == "__main__":
    main()
