import argparse
import copy
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from p3d3_common import answer_mean_nll, file_sha256, load_receiver, pack_answer, seed_everything, write_json, write_jsonl
from p3e_a_common import NativeHeadwiseReader
from p3e_c1_common import LearnableCanonicalHeadReader
from p3e_c2_common import head_diversity_loss, load_writer, writer_memory
from train_p3e_c2_writer import functional_alignment
from p3e_l_common import ConditionedNativeCache, condition_payload, native_memory


def freeze(module):
    module.requires_grad_(False)
    module.eval()
    return module


def traced_answer(model, tokenizer, reader, row, memory, max_length, device):
    ids, mask, labels = pack_answer(tokenizer, row, row["answer"], max_length, device)
    trace = {}
    with reader.inject(model, memory, trace):
        output = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
    return answer_mean_nll(output.logits, labels), trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--init-writer", required=True)
    parser.add_argument("--native-reader", required=True)
    parser.add_argument("--canonical-reader", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--depend-weight", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--route-weight", type=float, default=0.05)
    parser.add_argument("--readout-weight", type=float, default=0.10)
    parser.add_argument("--diversity-weight", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    cache = ConditionedNativeCache(args.memory)
    total = min(args.max_samples, len(cache))
    indices = list(range(total))
    model, tokenizer = load_receiver(args.model, device)
    freeze(model)

    native_checkpoint = torch.load(args.native_reader, map_location="cpu", weights_only=False)
    native_meta = native_checkpoint["reader_metadata"]
    native_reader = NativeHeadwiseReader(
        model, native_meta["selected_layers"], native_meta["rank"], native_meta["gate_init"]
    ).to(device)
    native_reader.load_state_dict(native_checkpoint["reader"])
    freeze(native_reader)

    canonical_checkpoint = torch.load(args.canonical_reader, map_location="cpu", weights_only=False)
    canonical_meta = canonical_checkpoint["reader_metadata"]
    canonical_reader = LearnableCanonicalHeadReader(
        model,
        canonical_meta["selected_layers"],
        canonical_meta["rank"],
        canonical_meta["gate_init"],
        canonical_meta["top_k"],
        0.25,
    ).to(device)
    canonical_reader.load_state_dict(canonical_checkpoint["reader"])
    freeze(canonical_reader)

    writer, init_checkpoint = load_writer(args.init_writer, device)
    writer.train()
    optimizer = torch.optim.AdamW(writer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    expected = {id(p) for p in writer.parameters() if p.requires_grad}
    actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if expected != actual:
        raise RuntimeError("Optimizer must contain only Writer parameters")

    history = []
    best_loss = float("inf")
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        writer.temperature = 1.0 + (0.25 - 1.0) * (epoch - 1) / max(1, args.epochs - 1)
        order = indices.copy()
        random.Random(args.seed + epoch).shuffle(order)
        totals = {name: [] for name in ("loss", "correct", "wrong", "dependency", "route", "readout", "diversity")}
        writer.train()
        for index in tqdm(order, desc=f"p3e_l_conditioned_writer_epoch{epoch}"):
            bundle = cache.load(index)
            correct = condition_payload(bundle, "correct_question")
            wrong = condition_payload(bundle, "correct_question_hard_shuffled_evidence")
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                _, native_trace = traced_answer(
                    model, tokenizer, native_reader, bundle["row"], native_memory(correct, device), args.max_length, device
                )
            correct_memory = writer_memory(writer, correct, device)
            correct_nll, canonical_trace = traced_answer(
                model, tokenizer, canonical_reader, bundle["row"], correct_memory, args.max_length, device
            )
            wrong_nll, _ = traced_answer(
                model, tokenizer, canonical_reader, bundle["row"], writer_memory(writer, wrong, device), args.max_length, device
            )
            dependency = F.relu(args.margin + correct_nll - wrong_nll)
            route_loss, readout_loss = functional_alignment(
                native_reader, canonical_reader, native_trace, canonical_trace
            )
            diversity = head_diversity_loss(correct_memory)
            loss = (
                correct_nll
                + args.depend_weight * dependency
                + args.route_weight * route_loss
                + args.readout_weight * readout_loss
                + args.diversity_weight * diversity
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(writer.parameters(), 1.0)
            optimizer.step()
            if any(p.grad is not None for p in model.parameters()):
                raise RuntimeError("Gradient reached Receiver")
            if any(p.grad is not None for p in native_reader.parameters()) or any(
                p.grad is not None for p in canonical_reader.parameters()
            ):
                raise RuntimeError("Gradient reached frozen Reader")
            for name, value in (
                ("loss", loss),
                ("correct", correct_nll),
                ("wrong", wrong_nll),
                ("dependency", dependency),
                ("route", route_loss),
                ("readout", readout_loss),
                ("diversity", diversity),
            ):
                totals[name].append(float(value.detach()))
        record = {
            "epoch": epoch,
            "temperature": writer.temperature,
            **{f"train_{name}": sum(values) / len(values) for name, values in totals.items()},
        }
        history.append(record)
        write_jsonl(output / "training_history.jsonl", history)
        if record["train_loss"] < best_loss:
            best_loss = record["train_loss"]
            best_epoch = epoch
            state = copy.deepcopy({name: tensor.detach().cpu() for name, tensor in writer.state_dict().items()})
            torch.save(
                {
                    "writer": state,
                    "writer_metadata": writer.metadata(),
                    "epoch": epoch,
                    "train_loss": best_loss,
                    "initial_writer": args.init_writer,
                    "initial_writer_sha256": file_sha256(args.init_writer),
                    "receiver_frozen": True,
                    "reader_frozen": True,
                    "only_writer_updated": True,
                },
                output / "checkpoint_best.pt",
            )
    write_json(
        output / "TRAIN_SUCCESS.json",
        {
            "status": "complete",
            "experiment": "P3-E-L question-conditioned Writer",
            "samples": total,
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_train_loss": best_loss,
            "initialized_from_c2": args.init_writer,
            "only_writer_updated": True,
            "checkpoint": str(output / "checkpoint_best.pt"),
        },
    )


if __name__ == "__main__":
    main()
