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
from protocol import apply_rope, capture_decision, final_logits
from translator import NativeKVTranslator

LABELS = "ABCD"


def serialize(item):
    question = item["question"]["stem"].strip()
    choices = sorted(item["question"]["choices"], key=lambda x: x["label"])
    if [x["label"] for x in choices] != list(LABELS):
        raise RuntimeError(f"Unexpected labels: {item['idx']}")
    options = [x["text"].strip() for x in choices]
    gold = LABELS.index(item["answerKey"])

    old_prefix = "Candidate answers:\n" + "\n".join(
        f"{LABELS[i]}. {option}" for i, option in enumerate(options)) + "\n\n"
    old_suffix = (f"Question:\n{question}\n\nChoose the best answer. "
                  f"Reply with only one letter from A to D.\nAnswer:")

    qh = "Question:\n"; question_end = len(qh) + len(question)
    oh = "\n\nOptions:\n"; options_start = question_end + len(oh)
    options_text = "\n".join(f"{LABELS[i]}. {option}" for i, option in enumerate(options))
    options_end = options_start + len(options_text)
    new_body = qh + question + oh + options_text + "\n\n"
    new_full = new_body + "Answer:"
    new_sender = new_body + qh + question
    return {"id": str(item["idx"]), "question": question, "options": options,
            "gold_index": gold, "labels": list(LABELS),
            "old_prefix": old_prefix, "old_suffix": old_suffix, "old_full": old_prefix + old_suffix,
            "new_body": new_body, "new_full": new_full, "new_sender": new_sender,
            "new_question_prefix": new_body[:options_start],
            "new_receiver_answer": new_body[options_end:] + "Answer:",
            "new_options_char_span": [options_start, options_end]}


def stable_choice_ids(tok, full_prompt):
    plain = tok.encode(full_prompt, add_special_tokens=False); result = []
    for label in LABELS:
        continuation = tok.encode(full_prompt + " " + label, add_special_tokens=False)
        if continuation[:len(plain)] != plain or len(continuation) != len(plain) + 1:
            raise RuntimeError(f"Choice {label} is not a stable single token")
        result.append(continuation[-1])
    if len(set(result)) != 4: raise RuntimeError("Duplicate choice IDs")
    return result


def encode_old(tok, row):
    enc = lambda x: tok.encode(x, add_special_tokens=False)
    prefix_map = tok(row["old_prefix"], add_special_tokens=False, return_offsets_mapping=True)
    prefix_plain, suffix = list(prefix_map["input_ids"]), enc(row["old_suffix"])
    full_plain = enc(row["old_full"])
    if prefix_plain + suffix != full_plain: raise RuntimeError(f"Old boundary mismatch: {row['id']}")
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    return {"prefix": bos + prefix_plain, "suffix": suffix, "full": bos + full_plain,
            "choice_ids": stable_choice_ids(tok, row["old_full"])}


def encode_new(tok, row):
    enc = lambda x: tok.encode(x, add_special_tokens=False)
    mapping = tok(row["new_body"], add_special_tokens=False, return_offsets_mapping=True)
    body_plain, full_plain, sender_plain = list(mapping["input_ids"]), enc(row["new_full"]), enc(row["new_sender"])
    question_plain = enc(row["new_question_prefix"])
    if full_plain[:len(body_plain)] != body_plain or sender_plain[:len(body_plain)] != body_plain:
        raise RuntimeError(f"New sender boundary mismatch: {row['id']}")
    if full_plain[:len(question_plain)] != question_plain:
        raise RuntimeError(f"New question boundary mismatch: {row['id']}")
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    a, b = row["new_options_char_span"]
    option_indices = [len(bos) + i for i, (start, end) in enumerate(mapping["offset_mapping"])
                      if min(int(end), b) > max(int(start), a)]
    return {"body": bos + body_plain, "full": bos + full_plain, "sender": bos + sender_plain,
            "question_prefix": bos + question_plain, "receiver_answer": enc(row["new_receiver_answer"]),
            "choice_ids": stable_choice_ids(tok, row["new_full"]), "option_indices": option_indices}


def dataset_rows(cfg, mode):
    raw = json.loads(Path(cfg["dataset"]).read_text(encoding="utf-8"))
    count = 2 if mode == "smoke" else cfg["test_samples"]
    return [serialize(x) for x in random.Random(cfg["seed"]).sample(raw, count)]


@torch.no_grad()
def capture_prefix(model, full_ids, prefix_length, choice_ids):
    keys, values, handles = {}, {}, []
    def hook(index):
        def apply(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            key = module.k_proj(hidden).view(shape)
            if hasattr(module, "k_norm"): key = module.k_norm(key)
            value = module.v_proj(hidden).view(shape)
            keys[index], values[index] = key[0, :prefix_length].cpu(), value[0, :prefix_length].cpu()
        return apply
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(hook(i), with_kwargs=True))
    try:
        ids = torch.tensor([full_ids], device="cuda")
        out = model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                          position_ids=torch.arange(len(full_ids), device="cuda")[None], use_cache=False)
        logits = model.lm_head(out.last_hidden_state[:, -1])[0].float()
    finally:
        for handle in handles: handle.remove()
    return (torch.stack([keys[i] for i in range(len(keys))]),
            torch.stack([values[i] for i in range(len(values))]),
            logits[torch.tensor(choice_ids, device="cuda")].cpu())


@torch.no_grad()
def old_query_importance(model, prefix_ids, suffix_ids, pre_rope_key):
    prefix_length, full_ids, per_layer, handles = len(prefix_ids), prefix_ids + suffix_ids, {}, []
    def hook(index):
        def apply(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            query = module.q_proj(hidden[:, prefix_length:]).view(
                len(suffix_ids), module.config.num_attention_heads, module.head_dim)
            if hasattr(module, "q_norm"): query = module.q_norm(query)
            query = apply_rope(model, query, torch.arange(prefix_length, len(full_ids), device="cuda"))
            key = apply_rope(model, pre_rope_key[index].cuda(), torch.arange(prefix_length, device="cuda"))
            repeat = module.config.num_attention_heads // module.config.num_key_value_heads
            scores = torch.einsum("hjd,htd->hjt", query.permute(1, 0, 2).float(),
                                  key.permute(1, 0, 2).repeat_interleave(repeat, 0).float()) / math.sqrt(module.head_dim)
            per_layer[index] = torch.softmax(scores, -1).amax(1).mean(0).cpu()
        return apply
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_pre_hook(hook(i), with_kwargs=True))
    try:
        ids = torch.tensor([full_ids], device="cuda")
        model.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                    position_ids=torch.arange(len(full_ids), device="cuda")[None], use_cache=False)
    finally:
        for handle in handles: handle.remove()
    importance = torch.stack([per_layer[i] for i in range(len(per_layer))]).mean(0)
    importance[0] = 0; importance /= importance.sum().clamp_min(1e-12)
    return importance


def cache_path(mode, route, sample_id):
    return ROOT / "runs" / mode / "cache" / route / f"{sample_id}.pt"


@torch.no_grad()
def prepare_llama(cfg, mode, rows):
    tok, model = tokenizer(cfg["models"]["llama"]), load_model(cfg, "llama")
    masses = {"old_options_first": [], "new_posterior_question": []}
    try:
        for n, row in enumerate(rows, 1):
            old = encode_old(tok, row)
            ok, ov, old_full_logits = capture_prefix(model, old["full"], len(old["prefix"]), old["choice_ids"])
            importance = old_query_importance(model, old["prefix"], old["suffix"], ok)
            spans = token_spans(tok, row["old_prefix"], old["prefix"])
            anchors = build_anchors(row["old_prefix"], spans, spans, importance, cfg["regions"])
            indices = [a["source_index"] for a in anchors]
            save_tensor(cache_path(mode, "old_options_first", row["id"]), {
                "source_k": ok[:, indices], "source_v": ov[:, indices], "importance": importance,
                "anchors": anchors, "selected_indices": indices, "full_choice_logits": old_full_logits,
                "selected_mass": selected_mass(importance, indices)})
            masses["old_options_first"].append(selected_mass(importance, indices))

            new = encode_new(tok, row)
            nk, nv, importance, new_sender_logits = capture_decision(
                model, new["sender"], len(new["body"]), new["choice_ids"])
            spans = token_spans(tok, row["new_body"], new["body"])
            option_spans = [span for span in spans if span.index in set(new["option_indices"])]
            anchors = build_anchors(row["new_body"], option_spans, option_spans, importance,
                                    cfg["regions"], *row["new_options_char_span"])
            indices = [a["source_index"] for a in anchors]
            save_tensor(cache_path(mode, "new_posterior_question", row["id"]), {
                "source_k": nk[:, indices], "source_v": nv[:, indices], "importance": importance,
                "anchors": anchors, "selected_indices": indices, "sender_choice_logits": new_sender_logits,
                "selected_mass": selected_mass(importance, indices, new["option_indices"])})
            masses["new_posterior_question"].append(selected_mass(importance, indices, new["option_indices"]))
            if n % 16 == 0 or n == len(rows): log(f"Llama old/new routing: {n}/{len(rows)}")
    finally:
        del model; torch.cuda.empty_cache()
    return {name + "_mean_selected_mass": sum(values) / len(values) for name, values in masses.items()}


def load_module(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    module = NativeKVTranslator("full28_diagonal").cuda().eval()
    module.load_state_dict(payload["state"], strict=True)
    return module, payload


def new_checkpoint(cfg):
    summary = read_json(Path(cfg["new_experiment"]) / "runs/study/stage_b/training_summary.json")
    return Path(summary["best_accuracy"]["path"]), int(summary["best_accuracy"]["step"])


def choice_kl(student, teacher, temperature):
    return (F.kl_div(F.log_softmax(student.float() / temperature, -1),
                     F.log_softmax(teacher.float() / temperature, -1),
                     reduction="sum", log_target=True) * temperature ** 2).item()


def paired(first, second):
    both = sum(a and b for a, b in zip(first, second)); only_first = sum(a and not b for a, b in zip(first, second))
    only_second = sum(b and not a for a, b in zip(first, second)); neither = len(first)-both-only_first-only_second
    d = only_first + only_second
    p = 1.0 if not d else min(1.0, 2 * sum(math.comb(d, k) for k in range(min(only_first, only_second)+1)) / 2**d)
    return {"both_correct": both, "only_first_correct": only_first, "only_second_correct": only_second,
            "both_wrong": neither, "accuracy_difference_second_minus_first": (only_second-only_first)/len(first),
            "mcnemar_exact_two_sided_p": p}


@torch.no_grad()
def evaluate(cfg, mode, rows, audit):
    old_path = Path(cfg["old_checkpoint"]); new_path, new_step = new_checkpoint(cfg)
    old_module, old_payload = load_module(old_path); new_module, new_payload = load_module(new_path)
    old_step = int(old_payload.get("step", 256))
    log(f"Old Options-first checkpoint={old_path} step={old_step}")
    log(f"New posterior-Question checkpoint={new_path} step={new_step}")
    qtok, ltok, model = tokenizer(cfg["models"]["qwen"]), tokenizer(cfg["models"]["llama"]), load_model(cfg, "qwen")
    names = ("old_qwen_full_native", "old_oracle", "old_translated",
             "new_qwen_full_native", "new_oracle", "new_translated")
    totals = {name: {"correct": 0, "agreement": 0, "kl": 0.0} for name in names}
    correctness = {name: [] for name in names}; records = []
    try:
        for n, row in enumerate(rows, 1):
            conditions, route_info = {}, {}

            old_q = encode_old(qtok, row); old_l = encode_old(ltok, row)
            qk, qv, qfull = capture_prefix(model, old_q["full"], len(old_q["prefix"]), old_q["choice_ids"])
            source = torch.load(cache_path(mode, "old_options_first", row["id"]), map_location="cpu", weights_only=True)
            anchors = build_anchors(row["old_prefix"], token_spans(ltok, row["old_prefix"], old_l["prefix"]),
                                    token_spans(qtok, row["old_prefix"], old_q["prefix"]),
                                    source["importance"], cfg["regions"])
            target_indices = [a["target_index"] for a in anchors]
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pk, pv = old_module(source["source_k"].cuda()[None], source["source_v"].cuda()[None])
            ids = torch.tensor(old_q["choice_ids"], device="cuda")
            oracle_k = torch.cat((qk[:, :1].cuda(), qk[:, target_indices].cuda()), 1)
            oracle_v = torch.cat((qv[:, :1].cuda(), qv[:, target_indices].cuda()), 1)
            translated_k = torch.cat((qk[:, :1].cuda(), pk[0]), 1)
            translated_v = torch.cat((qv[:, :1].cuda(), pv[0]), 1)
            conditions["old_qwen_full_native"] = qfull.cuda()
            conditions["old_oracle"] = final_logits(model, old_q["suffix"], oracle_k, oracle_v,
                positions=torch.arange(33, device="cuda"), suffix_start=33)[ids]
            conditions["old_translated"] = final_logits(model, old_q["suffix"], translated_k, translated_v,
                positions=torch.arange(33, device="cuda"), suffix_start=33)[ids]
            route_info["old"] = {"selected_indices": source["selected_indices"],
                                 "selected_text": [a["source_text"] for a in source["anchors"]],
                                 "target_indices": target_indices}

            new_q = encode_new(qtok, row); new_l = encode_new(ltok, row)
            qk, qv, _, qfull = capture_decision(model, new_q["full"], len(new_q["body"]), new_q["choice_ids"])
            source = torch.load(cache_path(mode, "new_posterior_question", row["id"]), map_location="cpu", weights_only=True)
            ls = token_spans(ltok, row["new_body"], new_l["body"]); qs = token_spans(qtok, row["new_body"], new_q["body"])
            ls = [x for x in ls if x.index in set(new_l["option_indices"])]
            qs = [x for x in qs if x.index in set(new_q["option_indices"])]
            anchors = build_anchors(row["new_body"], ls, qs, source["importance"], cfg["regions"],
                                    *row["new_options_char_span"])
            target_indices = [a["target_index"] for a in anchors]
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pk, pv = new_module(source["source_k"].cuda()[None], source["source_v"].cuda()[None])
            question_len = len(new_q["question_prefix"]); ids = torch.tensor(new_q["choice_ids"], device="cuda")
            def new_read(memory_k, memory_v):
                key = torch.cat((qk[:, :question_len].cuda(), memory_k), 1)
                value = torch.cat((qv[:, :question_len].cuda(), memory_v), 1)
                return final_logits(model, new_q["receiver_answer"], key, value,
                    positions=torch.arange(key.shape[1], device="cuda"), suffix_start=key.shape[1])[ids]
            conditions["new_qwen_full_native"] = qfull.cuda()
            conditions["new_oracle"] = new_read(qk[:, target_indices].cuda(), qv[:, target_indices].cuda())
            conditions["new_translated"] = new_read(pk[0], pv[0])
            route_info["new"] = {"selected_indices": source["selected_indices"],
                                 "selected_text": [a["source_text"] for a in source["anchors"]],
                                 "target_indices": target_indices}

            item = {"id": row["id"], "question": row["question"], "options": row["options"],
                    "gold_index": row["gold_index"], "routes": route_info, "conditions": {}}
            for name, logits in conditions.items():
                prediction = int(logits.argmax()); correct = prediction == row["gold_index"]
                route = name.split("_")[0]; teacher = conditions[f"{route}_oracle"] if name.endswith("translated") else logits
                agreement = prediction == int(teacher.argmax()); kl = choice_kl(logits, teacher, cfg["temperature"])
                correctness[name].append(correct); totals[name]["correct"] += correct
                totals[name]["agreement"] += agreement; totals[name]["kl"] += kl
                item["conditions"][name] = {"prediction": prediction, "correct": correct,
                    "choice_logits": logits.float().cpu().tolist(), "oracle_agreement": agreement, "oracle_choice_kl": kl}
            records.append(item)
            if n % 16 == 0 or n == len(rows): log(f"Qwen paired evaluation: {n}/{len(rows)}")
    finally:
        del model, old_module, new_module; torch.cuda.empty_cache()
    metrics = {name: {"accuracy": x["correct"]/len(rows), "oracle_agreement": x["agreement"]/len(rows),
                      "oracle_choice_kl": x["kl"]/len(rows)} for name, x in totals.items()}
    result = {"status": "completed", "dataset": "AraDiCE OpenBookQA English test", "seed": cfg["seed"],
        "sample_count": len(rows), "sampling": "fixed random sample without replacement",
        "old_protocol": "Options-first: Candidate answers KV -> Receiver-native Question/instruction/Answer:",
        "new_protocol": "standard raw order Sender Q->O->repeated-Q; Receiver native-Q -> external Options KV -> Answer:",
        "old_checkpoint": {"path": str(old_path), "step": old_step},
        "new_checkpoint": {"path": str(new_path), "step": new_step}, "routing_audit": audit, "metrics": metrics,
        "paired_old_vs_new_translated": paired(correctness["old_translated"], correctness["new_translated"]),
        "paired_old_oracle_vs_new_oracle": paired(correctness["old_oracle"], correctness["new_oracle"]),
        "paired_old_translated_vs_old_oracle": paired(correctness["old_translated"], correctness["old_oracle"]),
        "paired_new_translated_vs_new_oracle": paired(correctness["new_translated"], correctness["new_oracle"])}
    root = ROOT / "runs" / mode / "results"; root.mkdir(parents=True, exist_ok=True)
    save_json(root / "comparison.json", result)
    with (root / "per_sample_metrics.jsonl").open("w", encoding="utf-8") as f:
        for item in records: f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=("smoke", "study"), default="study")
    parser.add_argument("--keep-cache", action="store_true"); args = parser.parse_args()
    cfg = read_json(ROOT / "config.json"); rows = dataset_rows(cfg, args.mode)
    log(f"Selected OpenBookQA samples={len(rows)} first={rows[0]['id']} last={rows[-1]['id']}")
    audit = prepare_llama(cfg, args.mode, rows); result = evaluate(cfg, args.mode, rows, audit)
    if not args.keep_cache: shutil.rmtree(ROOT / "runs" / args.mode / "cache", ignore_errors=True)
    log(f"COMPLETED old_options_first={result['metrics']['old_translated']['accuracy']:.4f} "
        f"new_posterior_question={result['metrics']['new_translated']['accuracy']:.4f}")


if __name__ == "__main__": main()
