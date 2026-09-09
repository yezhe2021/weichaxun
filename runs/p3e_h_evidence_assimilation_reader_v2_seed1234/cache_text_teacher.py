import argparse
import hashlib
from pathlib import Path

import torch
from tqdm import tqdm

from p3d3_common import apply_chat, evidence_block, load_jsonl, load_receiver
from p3e_f_common import sha256_file, sha256_tensor, write_json


def full_evidence_prompt(tokenizer, row):
    system = "Answer the question using the supplied gold evidence. Give a short answer. End with exactly FINAL: <answer>."
    user = f"QUESTION\n{row['question']}\n\nGOLD EVIDENCE\n{evidence_block(row)}"
    return apply_chat(tokenizer, system, user) + "FINAL:"


def pack_teacher(tokenizer, row, max_length, device):
    prompt_ids = tokenizer(full_evidence_prompt(tokenizer, row), add_special_tokens=False).input_ids
    suffix = tokenizer(" " + row["answer"] + (tokenizer.eos_token or ""),
                       add_special_tokens=False).input_ids
    if len(prompt_ids) + len(suffix) > max_length:
        raise RuntimeError(f"Full-evidence teacher input exceeds {max_length}: {row['id']}")
    ids = torch.tensor([prompt_ids + suffix], dtype=torch.long, device=device)
    labels = ids.clone()
    labels[:, :len(prompt_ids)] = -100
    return ids, torch.ones_like(ids), labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.out)
    files = output / "files"
    files.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.data)[:args.max_samples]
    device = torch.device(args.device)
    model, tokenizer = load_receiver(args.model, device)
    entries = []
    with torch.inference_mode():
        for index, row in enumerate(tqdm(rows, desc="p3e_h_cache_full_text_teacher")):
            destination = files / f"sample_{index:05d}.pt"
            if destination.exists():
                payload = torch.load(destination, map_location="cpu", weights_only=False)
            else:
                ids, attention_mask, labels = pack_teacher(
                    tokenizer, row, args.max_length, device
                )
                output_state = model(input_ids=ids, attention_mask=attention_mask,
                                     use_cache=False, return_dict=True)
                shifted_labels = labels[:, 1:]
                selected = shifted_labels != -100
                answer_logits = output_state.logits[:, :-1].float()[selected]
                answer_ids = shifted_labels[selected]
                top_values, top_indices = answer_logits.topk(args.top_k, dim=-1)
                payload = {
                    "id": row["id"], "answer_token_ids": answer_ids.cpu(),
                    "topk_indices": top_indices.int().cpu(),
                    "topk_logits": top_values.half().cpu(),
                    "answer_tokens": int(answer_ids.numel()),
                }
                torch.save(payload, destination)
            digest = hashlib.sha256()
            for name in ("answer_token_ids", "topk_indices", "topk_logits"):
                digest.update(sha256_tensor(payload[name]).encode())
            entries.append({"index": index, "id": row["id"], "file": destination.name,
                            "answer_tokens": payload["answer_tokens"],
                            "tensor_sha256": digest.hexdigest()})
    write_json(output / "index.json", {
        "status": "complete", "samples": len(entries), "entries": entries,
        "teacher_model": args.model,
        "teacher_model_config_sha256": sha256_file(Path(args.model) / "config.json"),
        "teacher_condition": "question_plus_complete_gold_evidence",
        "distribution": f"answer_positions_top{args.top_k}_raw_logits",
        "max_length": args.max_length, "truncation": False,
    })
    write_json(output / "SUCCESS.json", {
        "status": "complete", "samples": len(entries), "top_k": args.top_k,
        "answer_positions_only": True, "prompt_or_padding_positions_cached": False,
        "truncation": False,
    })


if __name__ == "__main__":
    main()
