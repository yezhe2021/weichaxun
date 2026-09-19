import argparse
import copy
import random
from pathlib import Path

import torch
from tqdm import tqdm

from p3d3_common import forward_answer, load_receiver, seed_everything, write_json, write_jsonl
from p3e_a_common import NativeHeadwiseReader, native_memory_to
from p3e_n_common import ReceiverNativeConditionedCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--initialization", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.epochs != 5:
        raise RuntimeError("Reader C must use exactly the P3-E-M five-epoch protocol")

    seed_everything(args.seed)
    device = torch.device(args.device)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    cache = ReceiverNativeConditionedCache(args.memory)
    total = min(args.max_samples, len(cache))
    indices = list(range(total))
    model, tokenizer = load_receiver(args.model, device)
    initial = torch.load(args.initialization, map_location="cpu", weights_only=False)
    metadata = initial["reader_metadata"]
    reader = NativeHeadwiseReader(
        model,
        metadata["selected_layers"],
        metadata["rank"],
        metadata["gate_init"],
    ).to(device)
    reader.load_state_dict(initial["reader"])
    optimizer = torch.optim.AdamW(
        reader.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    expected = {id(parameter) for parameter in reader.parameters()}
    actual = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if expected != actual:
        raise RuntimeError("Optimizer must contain exactly Reader C parameters")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Qwen3-4B Receiver backbone is not frozen")

    history = []
    for epoch in range(1, args.epochs + 1):
        order = indices.copy()
        random.Random(args.seed + epoch).shuffle(order)
        reader.train()
        losses = []
        for index in tqdm(order, desc=f"p3e_n_reader_c_epoch{epoch}"):
            payload = cache.correct(index)
            optimizer.zero_grad(set_to_none=True)
            loss = forward_answer(
                model,
                tokenizer,
                reader,
                payload["row"],
                native_memory_to(payload, device),
                args.max_length,
                device,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
            optimizer.step()
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise RuntimeError("Gradient reached frozen Receiver")
            losses.append(float(loss.detach()))
        history.append(
            {
                "epoch": epoch,
                "train_answer_mean_nll": sum(losses) / len(losses),
                "gates": reader.gates().detach().cpu().tolist(),
            }
        )
        write_jsonl(output / "training_history.jsonl", history)
        state = copy.deepcopy(
            {name: tensor.detach().cpu() for name, tensor in reader.state_dict().items()}
        )
        torch.save(
            {
                "reader": state,
                "reader_metadata": reader.metadata(),
                "source": "Qwen3-4B Q-conditioned Native KV",
                "epoch": epoch,
                "args": vars(args),
                "initialization": args.initialization,
                "receiver_backbone_frozen": True,
                "writer_loaded": False,
                "canonical_projection_used": False,
            },
            output / f"checkpoint_epoch_{epoch:03d}.pt",
        )
    write_json(
        output / "TRAIN_SUCCESS.json",
        {
            "status": "complete",
            "experiment": "P3-E-N same-model Fresh Reader C",
            "samples": total,
            "epochs": 5,
            "loss": "answer-token mean NLL only",
            "checkpoint_selection": "fixed final epoch 5",
            "checkpoint": str(output / "checkpoint_epoch_005.pt"),
            "initialization": args.initialization,
            "trainable_parameters": sum(p.numel() for p in reader.parameters()),
            "receiver_backbone_frozen": True,
            "writer_loaded": False,
            "canonical_projection_used": False,
        },
    )


if __name__ == "__main__":
    main()
