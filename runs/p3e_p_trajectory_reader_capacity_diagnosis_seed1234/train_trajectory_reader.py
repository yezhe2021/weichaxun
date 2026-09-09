import argparse
import copy
import random
from pathlib import Path

import torch
from tqdm import tqdm

from p3d3_common import load_receiver, seed_everything, write_json, write_jsonl
from p3e_n_common import ReceiverNativeConditionedCache
from p3e_p_common import (
    MODES,
    SELECTED_LAYERS,
    TeacherTrajectoryCache,
    TrajectoryReader,
    answer_mean_nll,
    build_position_ids,
    native_memory,
    normalized_state_loss,
    pack_student,
    stack_trace,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    seed_everything(args.seed)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cache = ReceiverNativeConditionedCache(args.memory)
    teacher_cache = TeacherTrajectoryCache(args.teacher_cache)
    total = min(args.max_samples, len(cache), len(teacher_cache))
    model, tokenizer = load_receiver(args.model, device)
    reader = TrajectoryReader(model, args.mode, SELECTED_LAYERS).to(device)
    trainable = [parameter for parameter in reader.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if optimizer_ids != {id(parameter) for parameter in trainable}:
        raise RuntimeError("Optimizer parameter audit failed")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Frozen Receiver has trainable parameters")

    history = []
    for epoch in range(1, args.epochs + 1):
        order = list(range(total))
        random.Random(args.seed + epoch).shuffle(order)
        totals = {"total": [], "answer": [], "state": []}
        reader.train()
        for step, index in enumerate(
            tqdm(order, desc=f"p3e_p_{args.mode}_epoch{epoch}")
        ):
            payload = cache.correct(index)
            row = payload["row"]
            teacher = teacher_cache.load(index)
            packed = pack_student(tokenizer, row, device)
            if packed["answer_token_ids"] != teacher["answer_token_ids"]:
                raise RuntimeError("Teacher/student answer-token mismatch")
            memory = native_memory(payload, device)
            trace = {}
            optimizer.zero_grad(set_to_none=True)
            with reader.inject(
                model,
                memory,
                packed["prediction_positions"],
                trace,
            ):
                result = model(
                    input_ids=packed["input_ids"],
                    attention_mask=packed["attention_mask"],
                    position_ids=packed["position_ids"],
                    use_cache=False,
                    return_dict=True,
                )
            answer_loss = answer_mean_nll(result.logits, packed["labels"])
            corrected = stack_trace(trace, SELECTED_LAYERS, "corrected")
            target = teacher["teacher_states"].float().to(device)
            state_loss = normalized_state_loss(corrected, target)
            total_loss = (
                answer_loss
                if args.mode == "a0"
                else answer_loss + 0.5 * state_loss
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            if step == 0 and any(
                parameter.grad is not None for parameter in model.parameters()
            ):
                raise RuntimeError("Frozen Receiver received gradients")
            optimizer.step()
            totals["total"].append(float(total_loss.detach()))
            totals["answer"].append(float(answer_loss.detach()))
            totals["state"].append(float(state_loss.detach()))

        record = {
            "epoch": epoch,
            **{
                f"train_{key}_loss": sum(values) / len(values)
                for key, values in totals.items()
            },
        }
        history.append(record)
        write_jsonl(output / "training_history.jsonl", history)
        torch.save(
            {
                "reader": copy.deepcopy(
                    {
                        name: tensor.detach().cpu()
                        for name, tensor in reader.state_dict().items()
                    }
                ),
                "reader_metadata": reader.metadata(),
                "epoch": epoch,
                "losses": record,
                "receiver_frozen": True,
                "teacher_stopped_gradient": True,
            },
            output / f"checkpoint_epoch_{epoch:03d}.pt",
        )

    write_json(
        output / "TRAIN_SUCCESS.json",
        {
            "status": "complete",
            "mode": args.mode,
            "samples": total,
            "epochs": args.epochs,
            "last_checkpoint": str(output / f"checkpoint_epoch_{args.epochs:03d}.pt"),
            "loss": "answer" if args.mode == "a0" else "answer + 0.5 * normalized_state_mse",
            "receiver_frozen": True,
            "only_reader_updated": True,
        },
    )


if __name__ == "__main__":
    main()

