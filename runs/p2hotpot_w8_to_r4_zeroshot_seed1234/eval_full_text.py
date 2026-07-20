import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from hotpot_common import (
    answer_scores, extract_prediction, full_text_prompt, greedy_generate, load_jsonl,
    question_prompt, summarize, write_json, write_jsonl,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True); parser.add_argument("--condition", choices=("question_only", "full_text"), required=True)
    parser.add_argument("--out", required=True); parser.add_argument("--max-new-tokens", type=int, default=32); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); rows = load_jsonl(args.data); device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, trust_remote_code=True, local_files_only=True).to(device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    records = []
    for row in tqdm(rows, desc=f"hotpot_{args.model_name}_{args.condition}"):
        prompt = full_text_prompt(tokenizer, row) if args.condition == "full_text" else question_prompt(tokenizer, row)
        generated = greedy_generate(model, tokenizer, prompt, args.max_new_tokens)
        prediction, method = extract_prediction(generated["text"]); em, f1 = answer_scores(prediction, row["answer"])
        records.append({
            "id": row["id"], "condition": args.condition, "type": row["type"], "answer_type": row["answer_type"],
            "target": row["answer"], "prediction": prediction, "generated_text": generated["text"],
            "token_ids": generated["token_ids"], "eos_reached": generated["eos_reached"], "extraction_method": method,
            "em": em, "f1": f1, "source_em": em,
        })
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "per_sample_generation.jsonl", records)
    write_json(output / "SUCCESS.json", {"status": "complete", "model": args.model_name, "samples": len(rows), "conditions": summarize(records)})


if __name__ == "__main__":
    main()
