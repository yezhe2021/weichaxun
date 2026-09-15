import argparse
import hashlib
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from p3e_k_common import (
    read_json,
    read_jsonl,
    sentence_annotations,
    token_span_targets,
    write_json,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(root, entry):
    path = Path(entry["file"])
    return path if path.is_absolute() else root / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--native-index", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    native_index = read_json(args.native_index)
    native_root = Path(args.native_index).parent
    rows = read_jsonl(args.data)[:args.max_samples]
    entries = native_index["entries"][:len(rows)]
    if len(entries) != len(rows):
        raise RuntimeError("Native cache is shorter than requested sidecar")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16,
        trust_remote_code=True, local_files_only=True,
    ).to(args.device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    output_entries, counts = [], {"span": 0, "yes": 0, "no": 0, "span_unlocatable": 0}
    with torch.inference_mode():
        for index, row in enumerate(tqdm(rows, desc="p3e_k_cache_sidecar")):
            if entries[index]["id"] != row["id"]:
                raise RuntimeError(f"Native/data mismatch at {index}")
            native = torch.load(
                resolve(native_root, entries[index]),
                map_location="cpu", weights_only=False,
            )
            token_ids = torch.as_tensor(native["metadata"]["token_ids"], dtype=torch.long)
            reproduced = tokenizer(
                native["evidence"], add_special_tokens=True,
                truncation=True, max_length=1024,
            ).input_ids
            if reproduced != token_ids.tolist():
                raise RuntimeError("Qwen3-4B tokenizer does not match Native Evidence tokens")
            evidence_output = model.model(
                input_ids=token_ids.to(args.device)[None],
                attention_mask=torch.ones(1, token_ids.numel(), dtype=torch.long, device=args.device),
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[0].half().cpu()
            question_tokens = tokenizer(
                row["question"], return_tensors="pt", add_special_tokens=True,
            )
            question_tokens = {
                name: value.to(args.device) for name, value in question_tokens.items()
            }
            question = model.model(
                **question_tokens, use_cache=False, return_dict=True,
            ).last_hidden_state[0, -1].half().cpu()
            sentence_ids, sentence_keys, sentence_spans, gold_sentence_ids = sentence_annotations(
                native["evidence"], native["metadata"]["offsets"], row["supporting_facts"],
            )
            answer = str(row["answer"]).strip().lower()
            if answer in {"yes", "no"}:
                answer_kind, answer_spans = answer, []
            else:
                answer_spans = token_span_targets(
                    native["metadata"]["offsets"], row["answer_char_spans"],
                )
                answer_kind = "span" if answer_spans else "span_unlocatable"
            counts[answer_kind] += 1
            payload = {
                "id": row["id"],
                "question": question,
                "full_text_hidden": evidence_output,
                "sentence_ids": sentence_ids,
                "sentence_keys": sentence_keys,
                "sentence_char_spans": sentence_spans,
                "gold_sentence_ids": gold_sentence_ids,
                "answer_kind": answer_kind,
                "answer_spans": answer_spans,
            }
            destination = output / f"sample_{index:05d}.pt"
            torch.save(payload, destination)
            output_entries.append({
                "index": index,
                "id": row["id"],
                "file": destination.name,
                "tokens": int(token_ids.numel()),
                "answer_kind": answer_kind,
                "question_shape": list(question.shape),
                "full_text_shape": list(evidence_output.shape),
                "sha256": sha256_file(destination),
            })
    result = {
        "status": "complete",
        "experiment": "P3-E-K frozen Qwen3-4B sidecar cache",
        "samples": len(output_entries),
        "entries": output_entries,
        "answer_kind_counts": counts,
        "model": args.model,
        "native_index": args.native_index,
        "native_index_sha256": sha256_file(args.native_index),
        "data": args.data,
        "data_sha256": sha256_file(args.data),
        "question_representation": "frozen_Qwen3_4B_last_question_token_final_hidden",
        "full_text_representation": "frozen_Qwen3_4B_evidence_only_final_hidden_per_token",
        "sentence_ids": "deterministically_reconstructed_from_titles_numbered_sentences_and_offsets",
        "kv_recomputed": False,
    }
    write_json(output / "index.json", result)
    write_json(output / "SUCCESS.json", result)


if __name__ == "__main__":
    main()
