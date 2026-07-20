import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from p3a_common import answer_scores, extract_prediction, full_text_prompt, load_jsonl, question_prompt, summarize_records, write_json, write_jsonl


@torch.inference_mode()
def run(model, tokenizer, prompt, max_new_tokens):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False); encoded = {n: v.to(model.device) for n, v in encoded.items()}
    output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    tokens = output[0, encoded["input_ids"].shape[1]:].tolist()
    return tokenizer.decode(tokens, skip_special_tokens=True), tokens, tokenizer.eos_token_id in tokens


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model-name", required=True); parser.add_argument("--model", required=True); parser.add_argument("--data", required=True)
    parser.add_argument("--condition", choices=("question_only", "full_text"), required=True); parser.add_argument("--out", required=True); parser.add_argument("--max-new-tokens", type=int, default=32); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); rows = load_jsonl(args.data); tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, trust_remote_code=True, local_files_only=True).to(args.device).eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    records = []
    for row in tqdm(rows, desc=f"p3a_{args.model_name}_{args.condition}"):
        prompt = full_text_prompt(tokenizer, row) if args.condition == "full_text" else question_prompt(tokenizer, row)
        text, tokens, eos = run(model, tokenizer, prompt, args.max_new_tokens); prediction, method = extract_prediction(text); em, f1 = answer_scores(prediction, row["answer"])
        records.append({"id": row["id"], "condition": args.condition, "type": row["type"], "answer_type": row["answer_type"], "target": row["answer"], "prediction": prediction, "generated_text": text, "token_ids": tokens, "eos_reached": eos, "extraction_method": method, "em": em, "f1": f1, "source_em": em})
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True); write_jsonl(output / "per_sample_generation.jsonl", records)
    write_json(output / "SUCCESS.json", {"status": "complete", "model": args.model_name, "samples": len(rows), "conditions": summarize_records(records)})


if __name__ == "__main__": main()
