import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from p2iw_common import (
    PairCache, TokenCanonicalWriter, VariableAttentionProbe, label_vocabulary,
    projection, resolve_device, seed_everything, write_json, write_jsonl,
)


def teacher_state(row, projections, device):
    state = projection(row["hidden"].to(device), projections["hidden"], whiten=True)
    return F.layer_norm(state, (state.shape[-1],))


def one_pair(writer, probe, pair, label_to_id, projections, device, weights, margin):
    outputs, logits, targets, teachers = {}, {}, {}, {}
    content_terms = []
    for variant in ("base", "counterfactual"):
        row = pair[variant]
        output = writer(row["key_flat"].to(device), row["value_flat"].to(device))
        teacher = teacher_state(row, projections, device)
        content = (1.0 - F.cosine_similarity(output["shared"], teacher, dim=-1)).mean()
        content = content + F.mse_loss(F.layer_norm(output["shared"], (256,)), teacher)
        mask = torch.ones(1, output["keys"].shape[0], dtype=torch.bool, device=device)
        logit = probe(output["keys"][None], output["values"][None], mask)[0]
        target = label_to_id[row["answer"]]
        outputs[variant], logits[variant], targets[variant], teachers[variant] = output, logit, target, teacher
        content_terms.append(content)
    alignment = pair["_stable_alignment"].to(device)
    stable = F.mse_loss(
        outputs["base"]["shared"][alignment[:, 0]],
        outputs["counterfactual"]["shared"][alignment[:, 1]],
    )
    changed = {}
    for variant in ("base", "counterfactual"):
        answer_mask = pair[variant]["answer_mask"].to(device)
        if not answer_mask.any():
            raise RuntimeError("Answer span has no evidence tokens")
        changed[variant] = outputs[variant]["shared"][answer_mask].mean(0)
    changed_distance = 1.0 - F.cosine_similarity(changed["base"], changed["counterfactual"], dim=0)
    change = F.relu(float(margin) - changed_distance)
    classification = 0.5 * (
        F.cross_entropy(logits["base"][None], torch.tensor([targets["base"]], device=device))
        + F.cross_entropy(logits["counterfactual"][None], torch.tensor([targets["counterfactual"]], device=device))
    )
    switch = 0.5 * (
        F.relu(float(margin) + logits["base"][targets["counterfactual"]] - logits["base"][targets["base"]])
        + F.relu(float(margin) + logits["counterfactual"][targets["base"]] - logits["counterfactual"][targets["counterfactual"]])
    )
    loss = (
        weights["content"] * torch.stack(content_terms).mean()
        + weights["stable"] * stable + weights["change"] * change
        + weights["classification"] * classification + weights["switch"] * switch
    )
    stats = {
        "content": torch.stack(content_terms).mean(), "stable": stable, "change": change,
        "changed_distance": changed_distance, "classification": classification, "switch": switch,
    }
    pooled = torch.stack([
        outputs["base"]["shared"].mean(0), outputs["counterfactual"]["shared"].mean(0)
    ])
    return loss, stats, pooled, logits, targets


def batches(indices, size, seed):
    values = list(indices)
    random.Random(seed).shuffle(values)
    for start in range(0, len(values), size):
        yield values[start:start + size]


@torch.inference_mode()
def evaluate(writer, probe, cache, indices, label_to_id, projections, device, weights, margin):
    writer.eval(); probe.eval()
    losses, contents, correct, pair_correct = [], [], [], []
    for index in indices:
        loss, stats, _, logits, targets = one_pair(
            writer, probe, cache.load(index), label_to_id, projections, device, weights, margin
        )
        predictions = {variant: int(logits[variant].argmax()) for variant in ("base", "counterfactual")}
        local = [float(predictions[v] == targets[v]) for v in ("base", "counterfactual")]
        losses.append(float(loss.cpu())); contents.append(float(stats["content"].cpu()))
        correct.extend(local); pair_correct.append(local[0] * local[1])
    return {
        "loss": float(np.mean(losses)), "content_loss": float(np.mean(contents)),
        "accuracy": float(np.mean(correct)), "paired_consistency": float(np.mean(pair_correct)),
    }


@torch.inference_mode()
def small_gate(writer, probe, cache, indices, label_to_id, device):
    writer.eval(); probe.eval()
    variant_correct, pair_correct, pooled, rows = [], [], [], []
    for index in indices:
        pair = cache.load(index)
        local = []
        memory = {}
        for variant in ("base", "counterfactual"):
            row = pair[variant]
            output = writer(row["key_flat"].to(device), row["value_flat"].to(device))
            mask = torch.ones(1, output["keys"].shape[0], dtype=torch.bool, device=device)
            prediction = int(probe(output["keys"][None], output["values"][None], mask).argmax())
            target = label_to_id[row["answer"]]
            value = float(prediction == target)
            local.append(value); variant_correct.append(value)
            pooled.append(output["values"].mean(0))
            memory[variant] = (output, row)
            rows.append({"pair_id": row["pair_id"], "variant": variant, "target": row["answer"], "prediction_id": prediction, "correct": value})
        pair_correct.append(local[0] * local[1])
    matrix = F.normalize(torch.stack(pooled), dim=-1)
    cosine = matrix @ matrix.T
    offdiag = cosine[~torch.eye(len(matrix), dtype=torch.bool, device=device)]
    accuracy = float(np.mean(variant_correct)); paired = float(np.mean(pair_correct))
    return {
        "base_cf_accuracy": accuracy, "paired_consistency": paired,
        "state_swap_source_accuracy": accuracy,
        "mean_cross_sample_pooled_value_cosine": float(offdiag.mean().cpu()),
        "passed": bool(accuracy >= 0.95 and paired >= 0.90 and float(offdiag.mean().cpu()) < 0.998),
        "predictions": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--projections", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("small", "full"), required=True)
    parser.add_argument("--subset", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-pairs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--margin", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    cache = PairCache(args.index, capacity=max(8, args.batch_pairs + 2))
    if len(cache) != 512:
        raise ValueError("Writer training requires the exact 512-pair cache")
    labels = label_vocabulary(cache)
    label_to_id = {value: index for index, value in enumerate(labels)}
    bundle = torch.load(args.projections, map_location="cpu", weights_only=False)
    pca = bundle["pca"]
    writer_config = {"dim": 256, "rank": args.rank, "freeze_base": True}
    writer = TokenCanonicalWriter(pca, **writer_config).to(device)
    probe = VariableAttentionProbe(len(labels)).to(device)
    if args.mode == "small":
        train_indices = sorted(random.Random(args.seed).sample(range(448), args.subset))
        validation_indices = train_indices
        weights = {"content": 1.0, "stable": 0.0, "change": 0.0, "classification": 1.0, "switch": 0.2, "variance": 0.0}
    else:
        train_indices, validation_indices = list(range(448)), list(range(448, 512))
        weights = {"content": 1.0, "stable": 0.2, "change": 0.2, "classification": 0.1, "switch": 0.05, "variance": 0.01}
    parameters = [parameter for parameter in list(writer.parameters()) + list(probe.parameters()) if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    history, best, bad = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        writer.train(); probe.train(); epoch_rows = []
        for pair_batch in tqdm(batches(train_indices, args.batch_pairs, args.seed + epoch), desc=f"p2iw_{args.mode}_{epoch}", leave=False):
            pair_losses, pooled = [], []
            accum = {name: [] for name in ("content", "stable", "change", "changed_distance", "classification", "switch")}
            for index in pair_batch:
                loss, stats, representations, _, _ = one_pair(
                    writer, probe, cache.load(index), label_to_id, pca, device, weights, args.margin
                )
                pair_losses.append(loss); pooled.append(representations)
                for name in accum:
                    accum[name].append(stats[name])
            pooled = torch.cat(pooled, dim=0)
            std = torch.sqrt(pooled.var(dim=0, unbiased=False) + 1e-4)
            variance = F.relu(0.1 - std).mean()
            loss = torch.stack(pair_losses).mean() + weights["variance"] * variance
            optimizer.zero_grad(set_to_none=True); loss.backward()
            bad_gradients = [name for name, parameter in list(writer.named_parameters()) + [(f"probe.{n}", p) for n, p in probe.named_parameters()] if parameter.requires_grad and parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
            if bad_gradients:
                raise RuntimeError(f"Non-finite gradients: {bad_gradients[:5]}")
            torch.nn.utils.clip_grad_norm_(parameters, 1.0); optimizer.step()
            epoch_rows.append({"loss": float(loss.detach().cpu()), "variance": float(variance.detach().cpu()), **{name: float(torch.stack(values).mean().detach().cpu()) for name, values in accum.items()}})
        validation = evaluate(writer, probe, cache, validation_indices, label_to_id, pca, device, weights, args.margin)
        record = {"epoch": epoch, **{f"train_{key}": float(np.mean([row[key] for row in epoch_rows])) for key in epoch_rows[0]}, **{f"validation_{key}": value for key, value in validation.items()}}
        history.append(record); write_jsonl(output / "history.jsonl", history)
        criterion = validation["loss"]
        if criterion < best - 1e-5:
            best, bad = criterion, 0
            torch.save({
                "writer": writer.state_dict(), "probe": probe.state_dict(), "writer_config": writer_config,
                "labels": labels, "epoch": epoch, "mode": args.mode, "projection_file": str(Path(args.projections).resolve()),
            }, output / "best_checkpoint.pt")
        else:
            bad += 1
            if bad >= args.patience:
                break
    checkpoint = torch.load(output / "best_checkpoint.pt", map_location=device, weights_only=False)
    writer.load_state_dict(checkpoint["writer"]); probe.load_state_dict(checkpoint["probe"])
    final_validation = evaluate(writer, probe, cache, validation_indices, label_to_id, pca, device, weights, args.margin)
    result = {"status": "complete", "mode": args.mode, "best_epoch": checkpoint["epoch"], "validation": final_validation}
    if args.mode == "small":
        gate = small_gate(writer, probe, cache, train_indices, label_to_id, device)
        result["small_overfit_gate"] = {key: value for key, value in gate.items() if key != "predictions"}
        write_jsonl(output / "small_predictions.jsonl", gate["predictions"])
    write_json(output / "SUCCESS.json", result)


if __name__ == "__main__":
    main()
