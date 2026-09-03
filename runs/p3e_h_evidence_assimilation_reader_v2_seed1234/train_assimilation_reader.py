import argparse
import copy
import random
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from p3d3_common import (answer_mean_nll, answer_scores, generate, load_receiver,
                         pack_answer, seed_everything)
from p3e_f_common import CanonicalCache, memory_to, read_json, sha256_file, write_json, write_jsonl
from p3e_h_common import (EvidenceAssimilationReader, assert_boundaries,
                          assert_frozen_gradients, load_c1_reader)


class TeacherCache:
    def __init__(self, index_path, capacity=8):
        self.path = Path(index_path)
        self.root = self.path.parent
        self.entries = read_json(index_path)["entries"]
        self.capacity = int(capacity)
        self.loaded = OrderedDict()

    def load(self, index):
        if index not in self.loaded:
            path = self.root / self.entries[index]["file"]
            if not path.exists():
                path = self.root / "files" / self.entries[index]["file"]
            self.loaded[index] = torch.load(
                path, map_location="cpu", weights_only=False
            )
            while len(self.loaded) > self.capacity:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(index)
        return self.loaded[index]


def cpu_state(module):
    return copy.deepcopy({
        name: tensor.detach().cpu() for name, tensor in module.state_dict().items()
    })


def student_forward(model, tokenizer, reader, row, memory, max_length, device):
    ids, mask, labels = pack_answer(tokenizer, row, row["answer"], max_length, device)
    with reader.inject(model, memory):
        output = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True)
    shifted_labels = labels[:, 1:]
    selected = shifted_labels != -100
    answer_logits = output.logits[:, :-1].float()[selected]
    answer_ids = shifted_labels[selected]
    return answer_mean_nll(output.logits, labels), answer_logits, answer_ids


def teacher_kl(student_logits, student_ids, teacher, temperature, device):
    teacher_ids = teacher["answer_token_ids"].to(device=device, dtype=torch.long)
    if not torch.equal(student_ids, teacher_ids):
        raise RuntimeError("Teacher/student answer-token alignment mismatch")
    topk_indices = teacher["topk_indices"].to(device=device, dtype=torch.long)
    teacher_logits = teacher["topk_logits"].to(device=device, dtype=torch.float32)
    if topk_indices.shape[0] != student_logits.shape[0]:
        raise RuntimeError("Teacher/student answer length mismatch")
    teacher_log_prob = F.log_softmax(teacher_logits / temperature, dim=-1)
    teacher_prob = teacher_log_prob.exp()
    student_log_prob = F.log_softmax(student_logits / temperature, dim=-1)
    student_topk_log_prob = student_log_prob.gather(-1, topk_indices)
    return (teacher_prob * (teacher_log_prob - student_topk_log_prob)).sum(-1).mean() * (temperature ** 2)


@torch.inference_mode()
def initial_equivalence(model, tokenizer, old_reader, new_reader, payload, device, max_length):
    ids, mask, _ = pack_answer(tokenizer, payload["row"], payload["row"]["answer"], max_length, device)
    memory = memory_to(payload, device)
    with old_reader.inject(model, memory):
        old_logits = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).logits
    with new_reader.inject(model, memory):
        new_logits = model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).logits
    difference = (old_logits.float() - new_logits.float()).abs()
    return {"max_abs_logit_difference": float(difference.max()),
            "mean_abs_logit_difference": float(difference.mean())}


@torch.inference_mode()
def smoke_eval(model, tokenizer, reader, cache, negatives, count, device, max_new_tokens):
    rows = []
    for index in range(count):
        payload, wrong = cache.load(index), cache.load(negatives[index])
        correct = generate(model, tokenizer, reader, payload["row"], memory_to(payload, device), max_new_tokens)
        shuffled = generate(model, tokenizer, reader, payload["row"], memory_to(wrong, device), max_new_tokens)
        correct_em, correct_f1 = answer_scores(correct["prediction"], payload["row"]["answer"])
        _, shuffled_f1 = answer_scores(shuffled["prediction"], payload["row"]["answer"])
        rows.append({"id": payload["row"]["id"], "correct": correct, "shuffled": shuffled,
                     "correct_em": correct_em, "correct_f1": correct_f1,
                     "shuffled_current_f1": shuffled_f1})
    return {
        "correct_em": sum(row["correct_em"] for row in rows) / len(rows),
        "correct_f1": sum(row["correct_f1"] for row in rows) / len(rows),
        "shuffled_current_f1": sum(row["shuffled_current_f1"] for row in rows) / len(rows),
        "correct_shuffled_gap": sum(row["correct_f1"] - row["shuffled_current_f1"] for row in rows) / len(rows),
    }, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--memory-index", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--negatives", required=True)
    parser.add_argument("--teacher-index", required=True)
    parser.add_argument("--base-reader", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["smoke16", "formal512"], required=True)
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--bottleneck", type=int, default=128)
    parser.add_argument("--beta-init", type=float, default=0.01)
    parser.add_argument("--depend-weight", type=float, default=0.5)
    parser.add_argument("--kl-weight", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cache = CanonicalCache(args.memory_index, args.data, capacity=2)
    teachers = TeacherCache(args.teacher_index)
    negatives = read_json(args.negatives)["train512"]
    count = min(args.max_samples, 512)
    indices = list(range(count))
    model, tokenizer = load_receiver(args.model, device)
    c1_checkpoint = torch.load(args.base_reader, map_location="cpu", weights_only=False)
    old_reader = load_c1_reader(model, c1_checkpoint)
    reader = EvidenceAssimilationReader(
        model, c1_checkpoint, args.bottleneck, args.beta_init
    ).to(device)
    reader.set_temperature(0.25)
    equivalence = initial_equivalence(
        model, tokenizer, old_reader, reader, cache.load(0), device, args.max_length
    )
    if equivalence["max_abs_logit_difference"] > 1e-4:
        raise RuntimeError(f"Initial C1 equivalence failed: {equivalence}")
    del old_reader
    optimizer = torch.optim.AdamW(reader.new_parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    assert_boundaries(model, reader, optimizer)
    history, best_loss, best_epoch = [], float("inf"), -1
    for epoch in range(1, args.epochs + 1):
        order = indices.copy()
        random.Random(args.seed + epoch).shuffle(order)
        reader.train()
        losses, answers, dependencies, kls = [], [], [], []
        up_gradients, beta_gradients = [], []
        for index in tqdm(order, desc=f"p3e_h_{args.mode}_epoch{epoch}"):
            payload, wrong = cache.load(index), cache.load(negatives[index])
            teacher = teachers.load(index)
            if teacher["id"] != payload["row"]["id"]:
                raise RuntimeError("Teacher/data order mismatch")
            optimizer.zero_grad(set_to_none=True)
            correct_nll, student_logits, student_ids = student_forward(
                model, tokenizer, reader, payload["row"], memory_to(payload, device),
                args.max_length, device
            )
            shuffled_nll, _, _ = student_forward(
                model, tokenizer, reader, payload["row"], memory_to(wrong, device),
                args.max_length, device
            )
            dependency = F.relu(args.margin + correct_nll - shuffled_nll)
            kl = teacher_kl(student_logits, student_ids, teacher, args.temperature, device)
            loss = correct_nll + args.depend_weight * dependency + args.kl_weight * kl
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite Assimilation training loss")
            loss.backward()
            assert_frozen_gradients(model, reader)
            up_gradients.append(sum(float(adapter.up.weight.grad.float().norm())
                                    for adapter in reader.adapters if adapter.up.weight.grad is not None))
            beta_gradients.append(sum(float(adapter.beta_logit.grad.float().abs())
                                      for adapter in reader.adapters if adapter.beta_logit.grad is not None))
            torch.nn.utils.clip_grad_norm_(reader.new_parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
            answers.append(float(correct_nll.detach()))
            dependencies.append(float(dependency.detach()))
            kls.append(float(kl.detach()))
        epoch_loss = sum(losses) / len(losses)
        record = {
            "epoch": epoch, "total_loss": epoch_loss,
            "answer_nll": sum(answers) / len(answers),
            "depend_loss": sum(dependencies) / len(dependencies),
            "text_kl": sum(kls) / len(kls),
            "mean_up_gradient_norm": sum(up_gradients) / len(up_gradients),
            "mean_beta_gradient_abs": sum(beta_gradients) / len(beta_gradients),
            "betas": reader.betas().detach().cpu().tolist(),
            "old_gates": reader.old_gates().cpu().tolist(),
        }
        history.append(record)
        write_jsonl(output / "training_history.jsonl", history)
        state = {
            "reader": cpu_state(reader), "reader_metadata": reader.metadata(),
            "mode": args.mode, "epoch": epoch, "total_train_loss": epoch_loss,
            "args": vars(args), "base_reader": args.base_reader,
            "base_reader_sha256": sha256_file(args.base_reader),
            "teacher_index": args.teacher_index, "receiver_backbone_frozen": True,
            "c1_reader_frozen": True, "initial_equivalence": equivalence,
        }
        torch.save(state, output / "checkpoint_last.pt")
        if epoch_loss < best_loss:
            best_loss, best_epoch = epoch_loss, epoch
            torch.save(state, output / "checkpoint_best.pt")
    smoke_metrics = None
    if args.mode == "smoke16":
        reader.eval()
        smoke_metrics, rows = smoke_eval(
            model, tokenizer, reader, cache, negatives, count, device, args.max_new_tokens
        )
        write_jsonl(output / "smoke_generations.jsonl", rows)
    write_json(output / "TRAIN_SUCCESS.json", {
        "status": "complete", "experiment": "P3-E-H Evidence Assimilation Reader V2",
        "mode": args.mode, "samples": count, "epochs": args.epochs,
        "best_epoch_by_total_training_loss": best_epoch, "best_train_loss": best_loss,
        "initial_equivalence": equivalence, "smoke_free_running": smoke_metrics,
        "trainable_parameters": sum(parameter.numel() for parameter in reader.new_parameters()),
        "old_c1_reader_frozen": True, "receiver_backbone_frozen": True,
        "checkpoint_best": str(output / "checkpoint_best.pt"),
        "checkpoint_last": str(output / "checkpoint_last.pt"),
    })


if __name__ == "__main__":
    main()
