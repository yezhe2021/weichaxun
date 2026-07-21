import argparse
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from p3d_common import answer_scores, extract_prediction, full_text_prompt, load_receiver, question_prompt, read_json, write_json, write_jsonl


def load_rows(index_path):
    index_path = Path(index_path); index = read_json(index_path); rows = []
    for entry in index["entries"]:
        payload = torch.load(index_path.parent / entry["file"], map_location="cpu", weights_only=False)
        rows.append(payload["row"])
    return rows


def aggregate(records):
    result = {"n": len(records), "exact_match": sum(row["exact_match"] for row in records) / len(records), "f1": sum(row["f1"] for row in records) / len(records), "eos_rate": sum(row["eos_reached"] for row in records) / len(records)}
    grouped = defaultdict(list)
    for row in records: grouped[row["question_type"]].append(row)
    for kind in ("bridge", "comparison"):
        if grouped[kind]:
            result[kind] = {"n": len(grouped[kind]), "exact_match": sum(row["exact_match"] for row in grouped[kind]) / len(grouped[kind]), "f1": sum(row["f1"] for row in grouped[kind]) / len(grouped[kind])}
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model-name", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--cache", required=True); parser.add_argument("--condition", choices=("question_only", "full_text"), required=True)
    parser.add_argument("--out", required=True); parser.add_argument("--max-new-tokens", type=int, default=32); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); rows = load_rows(args.cache); model, tokenizer = load_receiver(args.model, torch.device(args.device)); records = []
    for index, row in enumerate(tqdm(rows, desc=f"p3d_{args.model_name}_{args.condition}")):
        prompt = full_text_prompt(tokenizer, row) if args.condition == "full_text" else question_prompt(tokenizer, row)
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False); encoded = {name: value.to(model.device) for name, value in encoded.items()}
        output = model.generate(**encoded, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        tokens = output[0, encoded["input_ids"].shape[1]:].tolist(); text = tokenizer.decode(tokens, skip_special_tokens=True)
        prediction, parse_status = extract_prediction(text); exact, f1 = answer_scores(prediction, row["answer"])
        records.append({"sample_index": index, "sample_id": row.get("id", str(index)), "condition": args.condition, "question_type": row.get("type", "unknown"), "question": row["question"], "gold_answer": row["answer"], "raw_generation": text, "prediction": prediction, "parse_status": parse_status, "exact_match": exact, "f1": f1, "eos_reached": float(tokenizer.eos_token_id in tokens), "generated_token_ids": tokens})
    output_dir = Path(args.out); output_dir.mkdir(parents=True, exist_ok=True); write_jsonl(output_dir / "per_sample_generation.jsonl", records)
    write_json(output_dir / "SUCCESS.json", {"status": "complete", "model": args.model_name, "condition": args.condition, "metrics": aggregate(records), "args": vars(args)})


if __name__ == "__main__": main()
