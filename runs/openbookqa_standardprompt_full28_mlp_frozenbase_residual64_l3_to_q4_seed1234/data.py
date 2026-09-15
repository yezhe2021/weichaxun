import json
import random
from pathlib import Path

import torch

from anchors import build_anchors, selected_mass
from common import digest, load_model, log, read_json, run_root, save_json, save_tensor, tokenizer
from offsets import token_spans
from protocol import capture_decision

LABELS = "ABCD"


def official_rows(cfg, split):
    path = Path(cfg["dataset"][split])
    raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    random.Random(cfg["seed"] + {"train": 0, "validation": 1, "test": 2}[split]).shuffle(raw)
    selected = raw[:cfg[f"{split}_samples"]]
    if len(selected) != cfg[f"{split}_samples"]:
        raise RuntimeError(f"Insufficient official OpenBookQA {split} rows: {len(selected)}")
    normalized = []
    for item in selected:
        choices = sorted(item["question"]["choices"], key=lambda value: value["label"])
        if [value["label"] for value in choices] != list(LABELS):
            raise RuntimeError(f"Unexpected OpenBookQA labels: {item['id']}")
        normalized.append({"id": str(item["id"]), "category": "openbookqa_official_main",
                           "question": item["question"]["stem"],
                           "options": [value["text"] for value in choices],
                           "gold_index": LABELS.index(item["answerKey"])})
    return normalized


def serialize(row):
    retained = [(index, str(option).strip()) for index, option in enumerate(row["options"])
                if str(option).strip() != "N/A"]
    old_gold = int(row["gold_index"])
    old_indices = [index for index, _ in retained]
    if old_gold not in old_indices: raise RuntimeError(f"Gold option removed as N/A: {row['id']}")
    gold = old_indices.index(old_gold)
    question = str(row["question"]).strip()
    question_header = "Question:\n"
    question_start = len(question_header)
    question_end = question_start + len(question)
    options_header = "\n\nOptions:\n"
    options_start = question_end + len(options_header)
    options_text = "\n".join(f"{LABELS[i]}. {option}" for i, (_, option) in enumerate(retained))
    options_end = options_start + len(options_text)
    body = question_header + question + options_header + options_text + "\n\n"
    answer_prefix = "Answer:"
    full_prompt = body + answer_prefix
    question_prefix = body[:options_start]
    receiver_answer = body[options_end:] + answer_prefix
    return {"body": body, "question_prefix": question_prefix,
            "receiver_answer": receiver_answer, "answer_prefix": answer_prefix,
            "full_prompt": full_prompt,
            "question_char_span": [question_start, question_end],
            "options_char_span": [options_start, options_end],
            "answer_prefix_char_span": [len(body), len(full_prompt)],
            "options": [option for _, option in retained], "gold_index": gold,
            "gold_label": LABELS[gold], "labels": list(LABELS[:len(retained)])}


def encode(tok, serialized):
    enc = lambda text: tok.encode(text, add_special_tokens=False)
    body_encoding = tok(serialized["body"], add_special_tokens=False, return_offsets_mapping=True)
    body_plain = list(body_encoding["input_ids"])
    answer_ids = enc(serialized["answer_prefix"])
    full_plain = enc(serialized["full_prompt"])
    question_prefix_plain = enc(serialized["question_prefix"])
    receiver_answer_ids = enc(serialized["receiver_answer"])
    if body_plain + answer_ids != full_plain:
        raise RuntimeError("Noncompositional body/Answer tokenizer boundary")
    if full_plain[:len(question_prefix_plain)] != question_prefix_plain:
        raise RuntimeError("Noncompositional native-question boundary")
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    body_ids, full_ids = bos + body_plain, bos + full_plain
    choice_ids = []
    for label in serialized["labels"]:
        continuation = enc(serialized["full_prompt"] + " " + label)
        if continuation[:len(full_plain)] != full_plain or len(continuation) != len(full_plain) + 1:
            raise RuntimeError(f"Choice {label} is not one stable continuation token")
        choice_ids.append(continuation[-1])
    if len(set(choice_ids)) != len(choice_ids): raise RuntimeError("Duplicate choice token IDs")
    question_prefix_ids = bos + question_prefix_plain
    option_start_char, option_end_char = serialized["options_char_span"]
    option_token_indices = [len(bos) + index for index, (start, end) in
                            enumerate(body_encoding["offset_mapping"])
                            if min(int(end), option_end_char) > max(int(start), option_start_char)]
    if not option_token_indices:
        raise RuntimeError("No option tokens found from offsets")
    return {"body": body_ids, "answer": answer_ids, "receiver_answer": receiver_answer_ids,
            "question_prefix": question_prefix_ids, "question_prefix_length": len(question_prefix_ids),
            "option_token_indices": option_token_indices, "full": full_ids,
            "answer_token_indices": list(range(len(body_ids), len(full_ids))),
            "decision_query_index": len(full_ids) - 1, "choice_ids": choice_ids}


def manifest_rows(cfg, split):
    payload = read_json(run_root(cfg) / "manifests" / f"{split}.json")
    if payload["signature"] != cfg["signature"]: raise RuntimeError("Manifest signature mismatch")
    return payload["rows"]


def prepare_manifests(cfg):
    toks = {family: tokenizer(path) for family, path in cfg["models"].items()}
    summary = {"signature": cfg["signature"], "dataset": "Official OpenBookQA Main splits",
               "prompt": "Question -> Options -> Answer:", "split_policy": cfg["split_policy"],
               "split_disjoint": True, "no_truncation": True, "splits": {},
               "answer_tokenization": {}, "observed_max_body_tokens": {"qwen": 0, "llama": 0}}
    split_ids = {}
    for split in ("train", "validation", "test"):
        old = official_rows(cfg, split); prepared = []
        for row in old:
            text = serialize(row)
            encoded = {family: encode(tok, text) for family, tok in toks.items()}
            for family, fields in encoded.items():
                summary["observed_max_body_tokens"][family] = max(
                    summary["observed_max_body_tokens"][family], len(fields["body"]))
            if any(len(fields["body"]) > cfg["max_body_tokens"] for fields in encoded.values()):
                lengths = {family: len(fields["body"]) for family, fields in encoded.items()}
                raise RuntimeError(f"Existing sample exceeds no-truncation safety ceiling: {row['id']} {lengths}")
            prepared.append({"id": row["id"], "category": row.get("category"),
                             "question": row["question"], **text, "encoded": encoded})
        if [row["id"] for row in prepared] != [row["id"] for row in old]:
            raise RuntimeError(f"ID/order drift in {split}")
        save_json(run_root(cfg) / "manifests" / f"{split}.json", {"signature": cfg["signature"], "rows": prepared})
        summary["splits"][split] = {"count": len(prepared), "ids_sha256": digest([row["id"] for row in prepared]),
                                    "first_id": prepared[0]["id"], "last_id": prepared[-1]["id"]}
        split_ids[split] = {row["id"] for row in prepared}
    if (split_ids["train"] & split_ids["validation"] or split_ids["train"] & split_ids["test"] or
            split_ids["validation"] & split_ids["test"]):
        raise RuntimeError("OpenBookQA train/validation/test overlap")
    if cfg["mode"] == "study" and sum(map(len, split_ids.values())) != sum(
            cfg[f"{split}_samples"] for split in ("train", "validation", "test")):
        raise RuntimeError("OpenBookQA split count mismatch")
    example = manifest_rows(cfg, "train")[0]
    for family, tok in toks.items():
        summary["answer_tokenization"][family] = {
            "ids": example["encoded"][family]["answer"],
            "tokens": tok.convert_ids_to_tokens(example["encoded"][family]["answer"]),
            "choice_ids": dict(zip(example["labels"], example["encoded"][family]["choice_ids"]))}
    save_json(run_root(cfg) / "audit" / "manifest_prompt_audit.json", summary)
    log(f"Prepared standard prompt with unchanged IDs: {summary['splits']}")
    return summary


def source_path(cfg, split, row):
    return run_root(cfg) / "source_selected_cache" / split / f"{row['id']}.pt"


def pair_path(cfg, split, row):
    return run_root(cfg) / "pair_cache" / split / f"{row['id']}.pt"


def load_source(cfg, split, row):
    payload = torch.load(source_path(cfg, split, row), map_location="cpu", weights_only=True)
    if payload["signature"] != cfg["signature"]: raise RuntimeError("Source cache signature mismatch")
    return payload


def load_pair(cfg, split, row):
    payload = torch.load(pair_path(cfg, split, row), map_location="cpu", weights_only=True)
    if payload["signature"] != cfg["signature"]: raise RuntimeError("Pair cache signature mismatch")
    return payload


@torch.no_grad()
def cache_llama(cfg):
    tok, model = tokenizer(cfg["models"]["llama"]), load_model(cfg, "llama")
    try:
        for split in ("train", "validation", "test"):
            selected_rows = manifest_rows(cfg, split)
            for number, row in enumerate(selected_rows, 1):
                path = source_path(cfg, split, row)
                if path.exists():
                    try:
                        load_source(cfg, split, row); continue
                    except RuntimeError: pass
                fields = row["encoded"]["llama"]
                key, value, importance, choice_logits = capture_decision(
                    model, fields["full"], len(fields["body"]), fields["choice_ids"])
                spans = token_spans(tok, row["body"], fields["body"])
                option_spans = [span for span in spans if span.index in set(fields["option_token_indices"])]
                anchors = build_anchors(row["body"], option_spans, option_spans, importance, cfg["regions"],
                                        *row["options_char_span"])
                indices = [0] + [anchor["source_index"] for anchor in anchors]
                save_tensor(path, {"signature": cfg["signature"], "id": row["id"],
                    "source_k": key[:, indices].contiguous(), "source_v": value[:, indices].contiguous(),
                    "importance": importance, "full_choice_logits": choice_logits,
                    "metadata": {"anchors": anchors, "selected_indices": indices[1:],
                                 "selected_attention_mass": selected_mass(
                                     importance, indices[1:], fields["option_token_indices"]),
                                 "option_token_indices": fields["option_token_indices"],
                                 "body_tokens": len(fields["body"]), "full_tokens": len(fields["full"]),
                                 "answer_token_indices": fields["answer_token_indices"],
                                 "decision_query_index": fields["decision_query_index"]}})
                if number % 32 == 0 or number == len(selected_rows):
                    log(f"Llama fresh decision cache {split}: {number}/{len(selected_rows)}")
    finally:
        del model; torch.cuda.empty_cache()


@torch.no_grad()
def cache_qwen_and_pairs(cfg):
    toks = {family: tokenizer(path) for family, path in cfg["models"].items()}
    model = load_model(cfg, "qwen")
    try:
        for split in ("train", "validation", "test"):
            selected_rows = manifest_rows(cfg, split)
            for number, row in enumerate(selected_rows, 1):
                path = pair_path(cfg, split, row)
                if path.exists():
                    try:
                        load_pair(cfg, split, row); continue
                    except RuntimeError: pass
                source = load_source(cfg, split, row)
                fields = row["encoded"]["qwen"]
                key, value, importance, choice_logits = capture_decision(
                    model, fields["full"], len(fields["body"]), fields["choice_ids"])
                spans = {family: token_spans(toks[family], row["body"], row["encoded"][family]["body"])
                         for family in ("llama", "qwen")}
                option_spans = {family: [span for span in spans[family]
                    if span.index in set(row["encoded"][family]["option_token_indices"])]
                    for family in ("llama", "qwen")}
                anchors = build_anchors(row["body"], option_spans["llama"], option_spans["qwen"],
                                        source["importance"], cfg["regions"], *row["options_char_span"])
                source_indices = [anchor["source_index"] for anchor in anchors]
                if source_indices != source["metadata"]["selected_indices"]:
                    raise RuntimeError(f"Llama selection changed while pairing: {row['id']}")
                target_indices = [0] + [anchor["target_index"] for anchor in anchors]
                self_anchors = build_anchors(row["body"], option_spans["qwen"], option_spans["qwen"],
                                             importance, cfg["regions"], *row["options_char_span"])
                self_indices = [0] + [anchor["source_index"] for anchor in self_anchors]
                payload = {"signature": cfg["signature"], "id": row["id"],
                    "target_k": key[:, target_indices].contiguous(),
                    "target_v": value[:, target_indices].contiguous(),
                    "question_k": key[:, :fields["question_prefix_length"]].contiguous(),
                    "question_v": value[:, :fields["question_prefix_length"]].contiguous(),
                    "qwen_full_choice_logits": choice_logits,
                    "llama_full_choice_logits": source["full_choice_logits"],
                    "metadata": {"anchors": anchors, "source_indices": [0] + source_indices,
                                 "target_indices": target_indices, "qwen_self_indices": self_indices,
                                 "qwen_self_anchors": self_anchors,
                                 "qwen_selected_attention_mass": selected_mass(
                                     importance, self_indices[1:], fields["option_token_indices"]),
                                 "qwen_body_tokens": len(fields["body"]), "qwen_full_tokens": len(fields["full"]),
                                 "qwen_answer_token_indices": fields["answer_token_indices"],
                                 "qwen_decision_query_index": fields["decision_query_index"]}}
                if split == "test":
                    payload["qwen_self_k"] = key[:, self_indices[1:]].contiguous()
                    payload["qwen_self_v"] = value[:, self_indices[1:]].contiguous()
                save_tensor(path, payload)
                if number % 32 == 0 or number == len(selected_rows):
                    log(f"Qwen fresh decision/pair cache {split}: {number}/{len(selected_rows)}")
    finally:
        del model; torch.cuda.empty_cache()


def phase0_audit(cfg):
    candidates = [(split, row) for split in ("train", "validation", "test")
                  for row in manifest_rows(cfg, split)]
    chosen = random.Random(cfg["seed"]).sample(candidates, min(cfg["audit_samples"], len(candidates)))
    records = []
    for split, row in chosen:
        source, pair = load_source(cfg, split, row), load_pair(cfg, split, row)
        anchors = pair["metadata"]["anchors"]
        record = {"sample_id": row["id"], "split": split, "raw_prompt": row["full_prompt"],
                  "question_char_span": row["question_char_span"], "options_char_span": row["options_char_span"],
                  "answer_prefix_char_span": row["answer_prefix_char_span"],
                  "llama_full_token_count": len(row["encoded"]["llama"]["full"]),
                  "qwen_full_token_count": len(row["encoded"]["qwen"]["full"]),
                  "llama_answer_token_indices": row["encoded"]["llama"]["answer_token_indices"],
                  "qwen_answer_token_indices": row["encoded"]["qwen"]["answer_token_indices"],
                  "decision_query_index": source["metadata"]["decision_query_index"],
                  "selectable_token_count": len(row["encoded"]["llama"]["option_token_indices"]),
                  "selected_32_indices": source["metadata"]["selected_indices"],
                  "selected_32_raw_text": [anchor["source_text"] for anchor in anchors],
                  "rawanchor_target_indices": pair["metadata"]["target_indices"][1:],
                  "rawanchor_target_raw_text": [anchor["target_text"] for anchor in anchors],
                  "anchor_overlap": [anchor["span_overlap"] for anchor in anchors],
                  "anchor_iou": [anchor["span_iou"] for anchor in anchors],
                  "source_selected_attention_mass": source["metadata"]["selected_attention_mass"]}
        if len(record["selected_32_indices"]) != 32 or len(record["rawanchor_target_indices"]) != 32:
            raise RuntimeError("Phase-0 memory count failure")
        if any(index in record["llama_answer_token_indices"] for index in record["selected_32_indices"]):
            raise RuntimeError("Answer token leaked into selection")
        option_indices = set(row["encoded"]["llama"]["option_token_indices"])
        if any(index not in option_indices for index in record["selected_32_indices"]):
            raise RuntimeError("Non-option token leaked into selection")
        records.append(record)
    root = run_root(cfg) / "audit"; root.mkdir(parents=True, exist_ok=True)
    with (root / "phase0_samples.jsonl").open("w", encoding="utf-8") as output:
        for record in records: output.write(json.dumps(record, ensure_ascii=False) + "\n")
    prompt_audit = read_json(root / "manifest_prompt_audit.json")
    summary = {"passed": True, "sample_count": len(records),
               "selection": "32 normalized option-text regions; max decision attention within each region",
               "query": "last Sender-native Answer: token", "candidates": "Options only; Question, position0 and Answer excluded",
               "answer_tokenization": prompt_audit["answer_tokenization"],
               "mean_anchor_iou": sum(sum(x["anchor_iou"]) for x in records) / (32 * len(records)),
               "mean_selected_attention_mass": sum(x["source_selected_attention_mass"] for x in records) / len(records)}
    save_json(root / "phase0_summary.json", summary)
    log(f"PHASE-0 AUDIT PASSED: {summary}")
    return summary
