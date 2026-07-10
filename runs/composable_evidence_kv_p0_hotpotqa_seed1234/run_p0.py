import argparse
import csv
import json
import random
import re
import string
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


CONDITIONS = ("question_only", "a_only", "b_only", "a_plus_b")


def normalize_answer(text):
    def remove_articles(value):
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value):
        table = str.maketrans("", "", string.punctuation)
        return value.translate(table)

    return " ".join(remove_articles(remove_punctuation(str(text).lower())).split())


def exact_match(prediction, gold):
    return float(normalize_answer(prediction) == normalize_answer(gold))


def answer_f1(prediction, gold):
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)
    common = sum(
        min(prediction_tokens.count(token), gold_tokens.count(token))
        for token in set(prediction_tokens) & set(gold_tokens)
    )
    if common == 0:
        return 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_answer(text, answer):
    normalized_answer = normalize_answer(answer)
    return float(bool(normalized_answer) and normalized_answer in normalize_answer(text))


def unique_support_titles(supporting_facts):
    titles = []
    for title, _ in supporting_facts:
        if title not in titles:
            titles.append(title)
    return titles


def make_document(title, sentences, source, supporting, supporting_sentences):
    text = " ".join(sentence.strip() for sentence in sentences if sentence.strip())
    return {
        "title": title,
        "text": text,
        "source": source,
        "supporting": bool(supporting),
        "supporting_sentences": list(supporting_sentences),
    }


def split_example(row, seed):
    context = {title: sentences for title, sentences in row.get("context", [])}
    support_titles = unique_support_titles(row.get("supporting_facts", []))
    if len(support_titles) != 2 or any(title not in context for title in support_titles):
        return None

    sample_rng = random.Random(f"{seed}:{row.get('_id', '')}")
    sample_rng.shuffle(support_titles)
    support_sentence_map = defaultdict(list)
    for title, sentence_id in row.get("supporting_facts", []):
        support_sentence_map[title].append(int(sentence_id))

    distractor_titles = [title for title in context if title not in support_titles]
    sample_rng.shuffle(distractor_titles)
    a_distractors = distractor_titles[0::2]
    b_distractors = distractor_titles[1::2]

    source_a = [
        make_document(
            support_titles[0],
            context[support_titles[0]],
            "A",
            True,
            support_sentence_map[support_titles[0]],
        )
    ]
    source_b = [
        make_document(
            support_titles[1],
            context[support_titles[1]],
            "B",
            True,
            support_sentence_map[support_titles[1]],
        )
    ]
    source_a.extend(make_document(title, context[title], "A", False, []) for title in a_distractors)
    source_b.extend(make_document(title, context[title], "B", False, []) for title in b_distractors)

    # Keep both gold documents before distractors so context trimming cannot drop one gold source first.
    combined = [source_a[0], source_b[0], *source_a[1:], *source_b[1:]]
    answer = str(row.get("answer", "")).strip()
    return {
        "id": row.get("_id", ""),
        "question": str(row.get("question", "")).strip(),
        "answer": answer,
        "type": row.get("type", ""),
        "level": row.get("level", ""),
        "support_titles": support_titles,
        "source_a": source_a,
        "source_b": source_b,
        "combined": combined,
        "answer_in_source_a": contains_answer(" ".join(doc["text"] for doc in source_a), answer),
        "answer_in_source_b": contains_answer(" ".join(doc["text"] for doc in source_b), answer),
    }


def load_examples(path, max_samples, seed):
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    random.Random(seed).shuffle(rows)
    examples = []
    rejected = 0
    for row in rows:
        example = split_example(row, seed)
        if example is None:
            rejected += 1
            continue
        if not example["question"] or not example["answer"]:
            rejected += 1
            continue
        examples.append(example)
        if max_samples > 0 and len(examples) >= max_samples:
            break
    if not examples:
        raise RuntimeError("No strict two-support-title HotpotQA examples were found")
    return examples, rejected


def documents_for_condition(example, condition):
    if condition == "question_only":
        return []
    if condition == "a_only":
        return example["source_a"]
    if condition == "b_only":
        return example["source_b"]
    if condition == "a_plus_b":
        return example["combined"]
    raise ValueError(f"Unknown condition: {condition}")


def format_evidence(documents):
    return "\n".join(
        f"[Source {document['source']} | {document['title']}] {document['text']}"
        for document in documents
    )


def render_prompt(tokenizer, question, evidence, prompt_style):
    evidence_block = evidence if evidence else "No external evidence is provided."
    user_text = (
        f"Question:\n{question}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        "Return only the short answer, without explanation."
    )
    if prompt_style == "chat" and tokenizer.chat_template:
        messages = [
            {
                "role": "system",
                "content": "Answer the question from the supplied evidence. Be concise and do not explain.",
            },
            {"role": "user", "content": user_text},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return (
        "Answer the question from the supplied evidence. Give only the short answer.\n\n"
        f"{user_text}\n\nAnswer:"
    )


def fit_prompt(tokenizer, question, evidence, prompt_style, max_input_tokens):
    prompt = render_prompt(tokenizer, question, evidence, prompt_style)
    token_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if len(token_ids) <= max_input_tokens:
        return prompt, len(token_ids), False, len(evidence)

    low = 0
    high = len(evidence)
    best = ""
    best_tokens = None
    while low <= high:
        midpoint = (low + high) // 2
        candidate_evidence = evidence[:midpoint]
        candidate = render_prompt(tokenizer, question, candidate_evidence, prompt_style)
        candidate_tokens = tokenizer(candidate, add_special_tokens=False).input_ids
        if len(candidate_tokens) <= max_input_tokens:
            best = candidate
            best_tokens = len(candidate_tokens)
            low = midpoint + 1
        else:
            high = midpoint - 1
    if best_tokens is None:
        raise RuntimeError("Question and generation template exceed max_input_tokens")
    return best, best_tokens, True, high


def extract_short_answer(text):
    text = str(text).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    for marker in ("\n\n", "\nQuestion:", "\nEvidence:", "<|im_end|>", "</s>"):
        if marker in text:
            text = text.split(marker, 1)[0]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0]
    for prefix in ("Answer:", "The answer is", "answer is", "It is", "It was"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
    return text.strip(" .\t\n\r\"'")


def parse_dtype(name, device):
    if name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    values = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = values[name]
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU inference requires --dtype float32 or auto")
    return dtype


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0])
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(examples):
    rows = []
    for sample_index, example in enumerate(examples):
        rows.append(
            {
                "sample": sample_index,
                "id": example["id"],
                "question": example["question"],
                "gold_answer": example["answer"],
                "type": example["type"],
                "level": example["level"],
                "support_titles": example["support_titles"],
                "source_a_titles": [doc["title"] for doc in example["source_a"]],
                "source_b_titles": [doc["title"] for doc in example["source_b"]],
                "answer_in_source_a": example["answer_in_source_a"],
                "answer_in_source_b": example["answer_in_source_b"],
            }
        )
    return rows


def aggregate_results(records, examples, conditions):
    condition_rows = []
    for condition in conditions:
        selected = [row for row in records if row["condition"] == condition]
        condition_rows.append(
            {
                "condition": condition,
                "n": len(selected),
                "exact_match": float(np.mean([row["exact_match"] for row in selected])),
                "answer_f1": float(np.mean([row["answer_f1"] for row in selected])),
                "contains_gold": float(np.mean([row["contains_gold"] for row in selected])),
                "mean_generated_tokens": float(np.mean([row["generated_tokens"] for row in selected])),
                "mean_input_tokens": float(np.mean([row["input_tokens"] for row in selected])),
                "truncated_rate": float(np.mean([row["input_was_truncated"] for row in selected])),
                "mean_latency_seconds": float(np.mean([row["latency_seconds"] for row in selected])),
            }
        )

    paired = {}
    required = set(CONDITIONS)
    if required.issubset(conditions):
        by_sample = defaultdict(dict)
        for row in records:
            by_sample[row["sample"]][row["condition"]] = row
        complete = [values for values in by_sample.values() if required.issubset(values)]
        ab_gain = [
            values["a_plus_b"]["answer_f1"]
            - max(values["a_only"]["answer_f1"], values["b_only"]["answer_f1"])
            for values in complete
        ]
        paired = {
            "n": len(complete),
            "mean_ab_f1_gain_over_best_single": float(np.mean(ab_gain)),
            "ab_beats_both_single_f1_rate": float(np.mean([gain > 0 for gain in ab_gain])),
            "compositional_exact_match_rate": float(
                np.mean(
                    [
                        values["a_plus_b"]["exact_match"] == 1
                        and values["a_only"]["exact_match"] == 0
                        and values["b_only"]["exact_match"] == 0
                        for values in complete
                    ]
                )
            ),
            "ab_f1_gain_over_question_only": float(
                np.mean(
                    [
                        values["a_plus_b"]["answer_f1"]
                        - values["question_only"]["answer_f1"]
                        for values in complete
                    ]
                )
            ),
            "answer_literal_in_source_a_rate": float(
                np.mean([example["answer_in_source_a"] for example in examples])
            ),
            "answer_literal_in_source_b_rate": float(
                np.mean([example["answer_in_source_b"] for example in examples])
            ),
        }
    return condition_rows, paired


def main():
    parser = argparse.ArgumentParser(description="P0 native-text baselines for composable Evidence-KV")
    parser.add_argument("--model", default="/home/yezhe/all_models/hub/models/Qwen/Qwen3-1___7B")
    parser.add_argument("--data", default="/home/yezhe/数据集/HotpotQA/raw/hotpot_dev_distractor_v1.json")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"))
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--prompt-style", choices=("chat", "plain"), default="chat")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    success_path = out_dir / "SUCCESS.json"
    if success_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite completed run: {success_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    examples, rejected = load_examples(args.data, args.max_samples, args.seed)
    manifest = build_manifest(examples)
    write_jsonl(out_dir / "manifest.jsonl", manifest)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.dry_run:
        previews = []
        for sample_index, example in enumerate(examples[:4]):
            for condition in args.conditions:
                evidence = format_evidence(documents_for_condition(example, condition))
                prompt, input_tokens, truncated, retained_chars = fit_prompt(
                    tokenizer,
                    example["question"],
                    evidence,
                    args.prompt_style,
                    args.max_input_tokens,
                )
                previews.append(
                    {
                        "sample": sample_index,
                        "id": example["id"],
                        "condition": condition,
                        "input_tokens": input_tokens,
                        "input_was_truncated": truncated,
                        "retained_evidence_chars": retained_chars,
                        "prompt": prompt,
                    }
                )
        write_jsonl(out_dir / "prompt_preview.jsonl", previews)
        with open(out_dir / "DRY_RUN.json", "w", encoding="utf-8") as handle:
            json.dump(
                {"status": "dry_run_complete", "n": len(examples), "rejected_before_limit": rejected},
                handle,
                indent=2,
                ensure_ascii=False,
            )
        return

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = parse_dtype(args.dtype, device)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    records = []
    progress = tqdm(total=len(examples) * len(args.conditions), desc="p0_free_running")
    with torch.inference_mode():
        for sample_index, example in enumerate(examples):
            for condition in args.conditions:
                evidence = format_evidence(documents_for_condition(example, condition))
                prompt, input_tokens, truncated, retained_chars = fit_prompt(
                    tokenizer,
                    example["question"],
                    evidence,
                    args.prompt_style,
                    args.max_input_tokens,
                )
                encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                encoded = {key: value.to(device) for key, value in encoded.items()}
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latency = time.perf_counter() - started
                generated_ids = generated[0, encoded["input_ids"].shape[1] :].detach().cpu().tolist()
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                prediction = extract_short_answer(generated_text)
                gold = example["answer"]
                records.append(
                    {
                        "sample": sample_index,
                        "id": example["id"],
                        "condition": condition,
                        "question": example["question"],
                        "gold_answer": gold,
                        "prediction": prediction,
                        "generated_text": generated_text,
                        "exact_match": exact_match(prediction, gold),
                        "answer_f1": answer_f1(prediction, gold),
                        "contains_gold": contains_answer(generated_text, gold),
                        "input_tokens": input_tokens,
                        "generated_tokens": len(generated_ids),
                        "input_was_truncated": float(truncated),
                        "retained_evidence_chars": retained_chars,
                        "latency_seconds": latency,
                    }
                )
                progress.update(1)
    progress.close()

    condition_summary, paired_summary = aggregate_results(records, examples, args.conditions)
    write_jsonl(out_dir / "per_sample_generation.jsonl", records)
    write_csv(out_dir / "condition_summary.csv", condition_summary)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"conditions": condition_summary, "paired": paired_summary},
            handle,
            indent=2,
            ensure_ascii=False,
        )
    with open(success_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "complete",
                "args": vars(args),
                "device": str(device),
                "resolved_dtype": str(dtype),
                "n": len(examples),
                "rejected_before_limit": rejected,
                "conditions": condition_summary,
                "paired": paired_summary,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
