import argparse
import copy
import random
from pathlib import Path

import torch
from tqdm import tqdm

from p3d3_common import forward_answer, load_receiver, seed_everything, write_json, write_jsonl
from p3e_a_common import NativeHeadwiseReader, native_memory_to
from p3e_m_common import FreshReaderMemory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-memory", required=True)
    parser.add_argument("--conditioned-memory", required=True)
    parser.add_argument("--initialization", required=True)
    parser.add_argument("--source", choices=["evidence_only", "question_conditioned"], required=True)
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
        raise RuntimeError("P3-E-M protocol requires exactly 5 epochs")
    seed_everything(args.seed)
    device = torch.device(args.device)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    cache = FreshReaderMemory(args.base_memory, args.conditioned_memory)
    total = min(args.max_samples, len(cache))
    indices = list(range(total))
    model, tokenizer = load_receiver(args.model, device)
    initialization = torch.load(args.initialization, map_location="cpu", weights_only=False)
    metadata = initialization["reader_metadata"]
    reader = NativeHeadwiseReader(
        model,
        metadata["selected_layers"],
        metadata["rank"],
        metadata["gate_init"],
    ).to(device)
    reader.load_state_dict(initialization["reader"])
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
        raise RuntimeError("Optimizer does not contain exactly Reader parameters")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Receiver backbone is not frozen")

    history = []
    for epoch in range(1, args.epochs + 1):
        order = indices.copy()
        random.Random(args.seed + epoch).shuffle(order)
        reader.train()
        losses = []
        for index in tqdm(order, desc=f"p3e_m_{args.source}_epoch{epoch}"):
            payload = (
                cache.evidence_only(index)
                if args.source == "evidence_only"
                else cache.question_conditioned(index)
            )
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
        record = {
            "epoch": epoch,
            "train_answer_mean_nll": sum(losses) / len(losses),
            "gates": reader.gates().detach().cpu().tolist(),
        }
        history.append(record)
        write_jsonl(output / "training_history.jsonl", history)
        state = copy.deepcopy(
            {name: tensor.detach().cpu() for name, tensor in reader.state_dict().items()}
        )
        torch.save(
            {
                "reader": state,
                "reader_metadata": reader.metadata(),
                "source": args.source,
                "epoch": epoch,
                "args": vars(args),
                "initialization": args.initialization,
                "receiver_backbone_frozen": True,
                "writer_loaded": False,
                "canonical_projection_used": False,
            },
            output / f"checkpoint_epoch_{epoch:03d}.pt",
        )
    final_checkpoint = output / "checkpoint_epoch_005.pt"
    write_json(
        output / "TRAIN_SUCCESS.json",
        {
            "status": "complete",
            "experiment": "P3-E-M Fresh Native Headwise Reader",
            "source": args.source,
            "samples": total,
            "epochs": args.epochs,
            "loss": "answer-token mean NLL only",
            "checkpoint_selection": "fixed final epoch 5",
            "checkpoint": str(final_checkpoint),
            "initialization": args.initialization,
            "trainable_parameters": sum(p.numel() for p in reader.parameters()),
            "receiver_backbone_frozen": True,
            "writer_loaded": False,
            "canonical_projection_used": False,
        },
    )


if __name__ == "__main__":
    main()
