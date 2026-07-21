import argparse
import json
import random
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from p3d_common import answer_scores, load_receiver, memory_to, read_json, resize_memory, write_json, write_jsonl
from p3d2_common import CONFIGURATIONS, GroundedEvidenceReader, canonical_cache, forward_grounded, generate_grounded, hard_negative_mapping


class TeacherCache:
    def __init__(self, index_path, capacity=4):
        self.path = Path(index_path); self.root = self.path.parent; self.index = read_json(index_path); self.entries = self.index["entries"]
        self.capacity = capacity; self.loaded = OrderedDict()
    def load(self, index):
        if index not in self.loaded:
            self.loaded[index] = torch.load(self.root / self.entries[index]["file"], map_location="cpu", weights_only=False)
            while len(self.loaded) > self.capacity: self.loaded.popitem(last=False)
        self.loaded.move_to_end(index); return self.loaded[index]


def align_count(labels, teacher):
    positions = (labels[0] != -100).nonzero(as_tuple=False).flatten()
    count = min(len(positions), int(teacher["target_ids"].numel()))
    return positions[:count], count


def grounding_loss(trace, positions, target_joint):
    joint = []
    for item in trace.values():
        attention = item["attention"][0].index_select(0, positions)
        router = item["router"][0].index_select(0, positions)
        joint.append((attention * router[..., None]).mean(dim=0))
    prediction = torch.stack(joint).mean(dim=0).clamp_min(1e-8)
    prediction = prediction / prediction.sum()
    target = target_joint.to(prediction.device).float(); target = target / target.sum().clamp_min(1e-8)
    return F.kl_div(prediction.log(), target, reduction="sum")


def execution_losses(trace, positions, teacher, student_logits):
    cosine, normalized_mse, norm = [], [], []
    teacher_delta = teacher["teacher_hidden_delta"].to(student_logits.device).float()
    q_rms = teacher["question_hidden_rms"].to(student_logits.device).float()
    for layer, item in trace.items():
        delta = item["delta"][0].index_select(0, positions).float()
        target = teacher_delta[layer, : len(positions)]
        cosine.append((1.0 - F.cosine_similarity(delta, target, dim=-1)).mean())
        normalized_mse.append(F.mse_loss(F.layer_norm(delta, (delta.shape[-1],)), F.layer_norm(target, (target.shape[-1],))))
        ratio = delta.square().mean(dim=-1).sqrt() / q_rms[layer, : len(positions)].clamp_min(1e-5)
        target_ratio = target.square().mean(dim=-1).sqrt() / q_rms[layer, : len(positions)].clamp_min(1e-5)
        norm.append(F.smooth_l1_loss(ratio, target_ratio))
    target_ids = teacher["target_ids"][: len(positions)].to(student_logits.device)
    student_gold = student_logits[: len(positions)].gather(-1, target_ids[:, None]).squeeze(-1)
    q_gold = teacher["question_gold_logits"][: len(positions)].to(student_logits.device).float()
    teacher_logit_delta = teacher["teacher_gold_logit_delta"][: len(positions)].to(student_logits.device).float()
    logit_delta = F.smooth_l1_loss((student_gold.float() - q_gold) / 5.0, teacher_logit_delta / 5.0)
    execution = torch.stack(cosine).mean() + 0.25 * torch.stack(normalized_mse).mean() + 0.25 * logit_delta
    return execution, torch.stack(norm).mean(), logit_delta


@torch.inference_mode()
def quick_validation(model, tokenizer, reader, cache, teachers, negative, device, limit, max_new_tokens):
    records, compatibility = [], []
    reader.eval()
    for index in range(min(limit, len(cache))):
        payload, wrong_payload = cache.load(index), cache.load(negative[index]); teacher = teachers.load(index)
        current = memory_to(payload, device); wrong = resize_memory(memory_to(wrong_payload, device), current["keys"].shape[1])
        q_hidden = teacher["question_prompt_hidden"].to(device).float()
        compatibility.append(float(reader.compatibility_from_hidden(q_hidden, current) > reader.compatibility_from_hidden(q_hidden, wrong)))
        for condition, memory in (("correct", current), ("shuffled", wrong)):
            result = generate_grounded(model, tokenizer, reader, payload["row"], memory, max_new_tokens, True)
            em, f1 = answer_scores(result["prediction"], payload["row"]["answer"])
            records.append({"condition": condition, "type": payload["row"].get("type", "unknown"), "em": em, "f1": f1})
    def mean(condition, field, kind=None):
        rows = [row for row in records if row["condition"] == condition and (kind is None or row["type"] == kind)]
        return sum(row[field] for row in rows) / max(1, len(rows))
    correct_f1, shuffled_f1 = mean("correct", "f1"), mean("shuffled", "f1")
    bridge_f1 = mean("correct", "f1", "bridge")
    compat = sum(compatibility) / max(1, len(compatibility))
    score = correct_f1 + 0.5 * bridge_f1 + 0.5 * (correct_f1 - shuffled_f1) + 0.25 * compat
    return {"correct_f1": correct_f1, "shuffled_f1": shuffled_f1, "bridge_f1": bridge_f1, "compatibility_accuracy": compat, "selection_score": score}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True); parser.add_argument("--protocol", required=True)
    parser.add_argument("--teacher-train", required=True); parser.add_argument("--teacher-validation", required=True)
    parser.add_argument("--configuration", choices=tuple(CONFIGURATIONS), required=True); parser.add_argument("--mode", choices=("small", "full"), required=True)
    parser.add_argument("--out", required=True); parser.add_argument("--init-checkpoint"); parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--small-samples", type=int, default=16); parser.add_argument("--validation-samples", type=int, default=32)
    parser.add_argument("--shared-blocks", type=int, default=4); parser.add_argument("--lr", type=float, default=2e-4); parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--lambda-ground", type=float, default=0.2); parser.add_argument("--lambda-exec", type=float, default=0.1)
    parser.add_argument("--lambda-compat", type=float, default=0.2); parser.add_argument("--lambda-norm", type=float, default=0.05); parser.add_argument("--compat-margin", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=1024); parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); random.seed(args.seed); torch.manual_seed(args.seed); device = torch.device(args.device); protocol = read_json(args.protocol)
    train_cache = canonical_cache(protocol, "train"); validation_cache = train_cache if args.mode == "small" else canonical_cache(protocol, "validation")
    train_teachers = TeacherCache(args.teacher_train); validation_teachers = train_teachers if args.mode == "small" else TeacherCache(args.teacher_validation)
    model, tokenizer = load_receiver(args.model, device); groups = int(protocol["canonical16"]["groups"]); dimension = int(protocol["canonical16"]["memory_dim"])
    reader = GroundedEvidenceReader(model, groups, dimension, CONFIGURATIONS[args.configuration], args.shared_blocks).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint["reader_metadata"] != reader.metadata(): raise RuntimeError("Reader init interface mismatch")
        reader.load_state_dict(checkpoint["reader"])
    if any(parameter.requires_grad for parameter in model.parameters()): raise RuntimeError("Receiver backbone is not frozen")
    trainable = list(reader.parameters()); optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    if {id(parameter) for group in optimizer.param_groups for parameter in group["params"]} != {id(parameter) for parameter in trainable}: raise RuntimeError("Optimizer is not Reader-only")
    train_limit = min(args.small_samples, len(train_cache)) if args.mode == "small" else len(train_cache)
    validation_limit = train_limit if args.mode == "small" else min(args.validation_samples, len(validation_cache))
    train_negative, validation_negative = hard_negative_mapping(train_cache), hard_negative_mapping(validation_cache)
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True); history = []; best_score = -float("inf"); best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        reader.train(); order = list(range(train_limit)); random.Random(args.seed + epoch).shuffle(order); optimizer.zero_grad(set_to_none=True); pending = 0
        for step, index in enumerate(tqdm(order, desc=f"p3d2_{args.configuration}_{args.mode}_e{epoch}"), 1):
            payload, wrong_payload, teacher = train_cache.load(index), train_cache.load(train_negative[index]), train_teachers.load(index)
            if str(teacher["row_id"]) != str(payload["row"].get("id", index)): raise RuntimeError("Teacher/cache sample mismatch")
            current = memory_to(payload, device); wrong = resize_memory(memory_to(wrong_payload, device), current["keys"].shape[1]); trace = {}
            answer_loss, student_logits, labels = forward_grounded(model, tokenizer, reader, payload["row"], current, payload["row"]["answer"], args.max_length, device, True, trace)
            positions, count = align_count(labels, teacher); positions = positions[:count]
            ground = grounding_loss(trace, positions, teacher["grounding_joint"])
            execution, norm, logit_delta = execution_losses(trace, positions, teacher, student_logits)
            q_hidden = teacher["question_prompt_hidden"].to(device).float()
            correct_score = reader.compatibility_from_hidden(q_hidden, current); wrong_score = reader.compatibility_from_hidden(q_hidden, wrong)
            compat = F.relu(args.compat_margin - correct_score + wrong_score)
            loss = answer_loss + args.lambda_ground * ground + args.lambda_exec * execution + args.lambda_compat * compat + args.lambda_norm * norm
            (loss / args.gradient_accumulation).backward(); pending += 1
            if any(parameter.grad is not None for parameter in model.parameters()): raise RuntimeError("Frozen receiver received gradients")
            if pending == args.gradient_accumulation or step == len(order):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True); pending = 0
            history.append({"epoch": epoch, "index": index, "answer": float(answer_loss.detach()), "ground": float(ground.detach()), "execution": float(execution.detach()), "compat": float(compat.detach()), "norm": float(norm.detach()), "gold_logit_delta": float(logit_delta.detach()), "total": float(loss.detach()), "gate_mean": float(reader.gates().detach().mean())})
        metrics = quick_validation(model, tokenizer, reader, validation_cache, validation_teachers, validation_negative, device, validation_limit, args.max_new_tokens)
        row = {"epoch": epoch, **metrics}; write_jsonl(output / "history.jsonl", history); write_jsonl(output / "validation_history.jsonl", [row] if epoch == 1 else [json.loads(line) for line in (output / "validation_history.jsonl").read_text(encoding="utf-8").splitlines()] + [row])
        checkpoint = {"format_version": 1, "configuration": args.configuration, "mode": args.mode, "reader": {name: tensor.detach().cpu() for name, tensor in reader.state_dict().items()}, "reader_metadata": reader.metadata(), "epoch": epoch, "validation": metrics, "only_reader_trainable": True, "args": vars(args)}
        torch.save(checkpoint, output / "checkpoint_latest.pt")
        if metrics["selection_score"] > best_score: best_score, best_epoch = metrics["selection_score"], epoch; torch.save(checkpoint, output / "checkpoint_best.pt")
    write_json(output / "TRAIN_SUCCESS.json", {"status": "complete", "configuration": args.configuration, "mode": args.mode, "best_epoch": best_epoch, "best_selection_score": best_score, "reader_parameters": sum(parameter.numel() for parameter in reader.parameters()), "receiver_parameters_updated": 0, "writer_parameters_updated": 0, "active_layers": CONFIGURATIONS[args.configuration], "shared_blocks": args.shared_blocks})


if __name__ == "__main__": main()
