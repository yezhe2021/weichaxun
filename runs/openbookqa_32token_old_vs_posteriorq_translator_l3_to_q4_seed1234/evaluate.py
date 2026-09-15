import argparse
import json
import math
import random
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

from anchors import build_anchors, selected_mass
from common import ROOT, load_model, log, read_json, save_json, save_tensor, tokenizer
from offsets import token_spans
from protocol import capture_decision, final_logits
from translator import NativeKVTranslator

LABELS = "ABCD"


def serialize(item):
    question = item["question"]["stem"].strip()
    choices = sorted(item["question"]["choices"], key=lambda x: x["label"])
    if [x["label"] for x in choices] != list(LABELS):
        raise RuntimeError(f"Unexpected labels for {item['idx']}")
    options = [x["text"].strip() for x in choices]
    gold = LABELS.index(item["answerKey"])
    qh = "Question:\n"
    question_end = len(qh) + len(question)
    oh = "\n\nOptions:\n"
    options_start = question_end + len(oh)
    options_text = "\n".join(f"{LABELS[i]}. {x}" for i, x in enumerate(options))
    options_end = options_start + len(options_text)
    body = qh + question + oh + options_text + "\n\n"
    answer = "Answer:"
    question_prefix = body[:options_start]
    receiver_answer = body[options_end:] + answer
    posterior = qh + question
    return {"id": str(item["idx"]), "question": question, "options": options,
            "gold_index": gold, "labels": list(LABELS), "body": body,
            "full_prompt": body + answer, "sender_prompt": body + posterior,
            "question_prefix": question_prefix, "receiver_answer": receiver_answer,
            "options_char_span": [options_start, options_end]}


def encode(tok, row):
    enc = lambda text: tok.encode(text, add_special_tokens=False)
    mapping = tok(row["body"], add_special_tokens=False, return_offsets_mapping=True)
    body_plain = list(mapping["input_ids"])
    full_plain, sender_plain = enc(row["full_prompt"]), enc(row["sender_prompt"])
    question_plain = enc(row["question_prefix"])
    if full_plain[:len(body_plain)] != body_plain or sender_plain[:len(body_plain)] != body_plain:
        raise RuntimeError(f"Non-compositional sender boundary: {row['id']}")
    if full_plain[:len(question_plain)] != question_plain:
        raise RuntimeError(f"Non-compositional question boundary: {row['id']}")
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    body, full, sender = bos + body_plain, bos + full_plain, bos + sender_plain
    choice_ids = []
    for label in row["labels"]:
        continuation = enc(row["full_prompt"] + " " + label)
        if continuation[:len(full_plain)] != full_plain or len(continuation) != len(full_plain) + 1:
            raise RuntimeError(f"Choice {label} not one token for {row['id']}")
        choice_ids.append(continuation[-1])
    a, b = row["options_char_span"]
    option_indices = [len(bos) + i for i, (start, end) in enumerate(mapping["offset_mapping"])
                      if min(int(end), b) > max(int(start), a)]
    return {"body": body, "full": full, "sender": sender,
            "question_prefix": bos + question_plain,
            "receiver_answer": enc(row["receiver_answer"]),
            "choice_ids": choice_ids, "option_indices": option_indices}


def rows(cfg, mode):
    raw = json.loads(Path(cfg["dataset"]).read_text(encoding="utf-8"))
    count = 2 if mode == "smoke" else cfg["test_samples"]
    chosen = random.Random(cfg["seed"]).sample(raw, count)
    return [serialize(x) for x in chosen]


def cache_path(mode, route, sample_id):
    return ROOT / "runs" / mode / "cache" / route / f"{sample_id}.pt"


def best_checkpoint(experiment):
    summary = read_json(Path(experiment) / "runs" / "study" / "stage_b" / "training_summary.json")
    item = summary["best_accuracy"]
    return Path(item["path"]), int(item["step"])


def load_translator(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    module = NativeKVTranslator(payload.get("architecture", "full28_diagonal")).cuda().eval()
    module.load_state_dict(payload["state"], strict=True)
    return module, payload


@torch.no_grad()
def prepare_llama(cfg, mode, data):
    tok, model = tokenizer(cfg["models"]["llama"]), load_model(cfg, "llama")
    audit = {"old_selected_mass": [], "new_selected_mass": [], "selection_overlap": []}
    try:
        for n, row in enumerate(data, 1):
            fields = encode(tok, row)
            spans = token_spans(tok, row["body"], fields["body"])
            option_spans = [span for span in spans if span.index in set(fields["option_indices"])]
            selections = {}
            for route, ids in (("old_answer_query", fields["full"]),
                               ("new_posterior_question", fields["sender"])):
                key, value, importance, choice_logits = capture_decision(
                    model, ids, len(fields["body"]), fields["choice_ids"])
                anchors = build_anchors(row["body"], option_spans, option_spans, importance, cfg["regions"],
                                        *row["options_char_span"])
                indices = [a["source_index"] for a in anchors]
                if len(indices) != 32 or any(i not in set(fields["option_indices"]) for i in indices):
                    raise RuntimeError(f"Selection audit failed: {row['id']} {route}")
                selections[route] = indices
                save_tensor(cache_path(mode, route, row["id"]), {
                    "source_k": key[:, indices].contiguous(),
                    "source_v": value[:, indices].contiguous(),
                    "importance": importance, "anchors": anchors,
                    "selected_indices": indices, "choice_logits": choice_logits,
                    "selected_mass": selected_mass(importance, indices, fields["option_indices"])})
            audit["old_selected_mass"].append(torch.load(cache_path(mode, "old_answer_query", row["id"]), weights_only=True)["selected_mass"])
            audit["new_selected_mass"].append(torch.load(cache_path(mode, "new_posterior_question", row["id"]), weights_only=True)["selected_mass"])
            audit["selection_overlap"].append(len(set(selections["old_answer_query"]) & set(selections["new_posterior_question"])) / 32)
            if n % 16 == 0 or n == len(data): log(f"Llama routing cache: {n}/{len(data)}")
    finally:
        del model; torch.cuda.empty_cache()
    return {k: sum(v) / len(v) for k, v in audit.items()}


def choice_kl(student, teacher, temperature=1.0):
    s = F.log_softmax(student.float() / temperature, -1)
    t = F.log_softmax(teacher.float() / temperature, -1)
    return F.kl_div(s, t, reduction="sum", log_target=True).item() * temperature ** 2


def receiver_logits(model, fields, question_k, question_v, memory_k, memory_v):
    key = torch.cat((question_k.cuda(), memory_k), 1)
    value = torch.cat((question_v.cuda(), memory_v), 1)
    return final_logits(model, fields["receiver_answer"], key, value,
                        positions=torch.arange(key.shape[1], device="cuda"), suffix_start=key.shape[1])


def paired(a, b):
    both = sum(x and y for x, y in zip(a, b)); only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b)); neither = len(a) - both - only_a - only_b
    discordant = only_a + only_b
    p = 1.0 if discordant == 0 else min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(only_a, only_b) + 1)) / (2 ** discordant))
    return {"both_correct": both, "only_first_correct": only_a, "only_second_correct": only_b,
            "both_wrong": neither, "accuracy_difference_second_minus_first": (only_b - only_a) / len(a),
            "mcnemar_exact_two_sided_p": p}


@torch.no_grad()
def evaluate(cfg, mode, data, routing_audit):
    old_path, old_step = best_checkpoint(cfg["old_experiment"])
    new_path, new_step = best_checkpoint(cfg["new_experiment"])
    log(f"Old checkpoint: {old_path} (step {old_step})")
    log(f"New checkpoint: {new_path} (step {new_step})")
    old, old_payload = load_translator(old_path)
    new, new_payload = load_translator(new_path)
    tok, llama_tok = tokenizer(cfg["models"]["qwen"]), tokenizer(cfg["models"]["llama"])
    model = load_model(cfg, "qwen")
    records = []
    correct = {k: [] for k in ("qwen_full_native", "old_oracle", "old_translated", "new_oracle", "new_translated")}
    totals = {k: {"correct": 0, "oracle_agreement": 0, "choice_kl": 0.0}
              for k in correct}
    try:
        for n, row in enumerate(data, 1):
            fields = encode(tok, row)
            qk, qv, _, full_choice = capture_decision(model, fields["full"], len(fields["body"]), fields["choice_ids"])
            qspans = token_spans(tok, row["body"], fields["body"])
            qwen_option_spans = [span for span in qspans if span.index in set(fields["option_indices"])]
            llama_fields = encode(llama_tok, row)
            llama_spans = token_spans(llama_tok, row["body"], llama_fields["body"])
            llama_option_spans = [span for span in llama_spans if span.index in set(llama_fields["option_indices"])]
            question_len = len(fields["question_prefix"])
            conditions = {"qwen_full_native": full_choice.cuda()}
            route_meta = {}
            for route, module, alias in (("old_answer_query", old, "old"),
                                         ("new_posterior_question", new, "new")):
                source = torch.load(cache_path(mode, route, row["id"]), map_location="cpu", weights_only=True)
                anchors = build_anchors(row["body"],
                                        llama_option_spans,
                                        qwen_option_spans, source["importance"], cfg["regions"], *row["options_char_span"])
                target_indices = [a["target_index"] for a in anchors]
                target_k, target_v = qk[:, target_indices].cuda(), qv[:, target_indices].cuda()
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    pred_k, pred_v = module(source["source_k"].cuda()[None], source["source_v"].cuda()[None])
                oracle_logits = receiver_logits(model, fields, qk[:, :question_len], qv[:, :question_len], target_k, target_v)
                translated_logits = receiver_logits(model, fields, qk[:, :question_len], qv[:, :question_len], pred_k[0], pred_v[0])
                ids = torch.tensor(fields["choice_ids"], device="cuda")
                conditions[f"{alias}_oracle"] = oracle_logits[ids]
                conditions[f"{alias}_translated"] = translated_logits[ids]
                route_meta[alias] = {"selected_indices": source["selected_indices"],
                                     "selected_text": [a["source_text"] for a in source["anchors"]],
                                     "target_indices": target_indices, "checkpoint_step": old_step if alias == "old" else new_step}
            gold = row["gold_index"]
            oracle_pred = {"old": int(conditions["old_oracle"].argmax()), "new": int(conditions["new_oracle"].argmax())}
            item = {"id": row["id"], "question": row["question"], "options": row["options"],
                    "gold_index": gold, "routes": route_meta, "conditions": {}}
            for name, logits in conditions.items():
                pred = int(logits.argmax()); is_correct = pred == gold
                teacher = conditions[name.split("_")[0] + "_oracle"] if name.endswith("translated") else logits
                agreement = pred == int(teacher.argmax())
                kl = choice_kl(logits, teacher, cfg["temperature"])
                correct[name].append(is_correct); totals[name]["correct"] += is_correct
                totals[name]["oracle_agreement"] += agreement; totals[name]["choice_kl"] += kl
                item["conditions"][name] = {"prediction": pred, "correct": is_correct,
                                              "choice_logits": logits.float().cpu().tolist(),
                                              "oracle_agreement": agreement, "oracle_choice_kl": kl}
            records.append(item)
            if n % 16 == 0 or n == len(data): log(f"Qwen OpenBookQA evaluation: {n}/{len(data)}")
    finally:
        del model, old, new; torch.cuda.empty_cache()
    metrics = {name: {"accuracy": x["correct"] / len(data),
                      "oracle_agreement": x["oracle_agreement"] / len(data),
                      "oracle_choice_kl": x["choice_kl"] / len(data)} for name, x in totals.items()}
    comparison = {
        "status": "completed", "dataset": "AraDiCE OpenBookQA English test", "seed": cfg["seed"],
        "sample_count": len(data), "sampling": "fixed random sample without replacement",
        "protocol_shared": "32 normalized Options regions; Full28-Diagonal; Receiver-native Question + external Options KV + native Answer:",
        "old": {"routing": "last Answer: token", "checkpoint": str(old_path), "checkpoint_step": old_step},
        "new": {"routing": "last token of repeated posterior Question", "checkpoint": str(new_path), "checkpoint_step": new_step},
        "routing_audit": routing_audit, "metrics": metrics,
        "paired_old_vs_new_translated": paired(correct["old_translated"], correct["new_translated"]),
        "paired_old_oracle_vs_new_oracle": paired(correct["old_oracle"], correct["new_oracle"]),
        "paired_old_translated_vs_old_oracle": paired(correct["old_translated"], correct["old_oracle"]),
        "paired_new_translated_vs_new_oracle": paired(correct["new_translated"], correct["new_oracle"])}
    root = ROOT / "runs" / mode / "results"; root.mkdir(parents=True, exist_ok=True)
    save_json(root / "comparison.json", comparison)
    with (root / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as f:
        for item in records: f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return comparison


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=["smoke", "study"], default="study")
    parser.add_argument("--keep-cache", action="store_true"); args = parser.parse_args()
    cfg = read_json(ROOT / "config.json"); data = rows(cfg, args.mode)
    log(f"OpenBookQA selected samples={len(data)} first={data[0]['id']} last={data[-1]['id']}")
    routing_audit = prepare_llama(cfg, args.mode, data)
    result = evaluate(cfg, args.mode, data, routing_audit)
    if not args.keep_cache: shutil.rmtree(ROOT / "runs" / args.mode / "cache", ignore_errors=True)
    log(f"COMPLETED old={result['metrics']['old_translated']['accuracy']:.4f} new={result['metrics']['new_translated']['accuracy']:.4f}")


if __name__ == "__main__": main()
