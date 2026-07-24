import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from p3d3_common import (
    aliases_overlap,
    evidence_block,
    file_sha256,
    normalize_answer,
    write_json,
)
from p3e_c2_common import SenderNativeHeadwiseCache
from p3e_l_common import CONDITIONS


class NativeCapture:
    def __init__(self, model, layers):
        self.model = model
        self.layers = list(layers)
        self.states = {}
        self.handles = []

    def __enter__(self):
        for layer_index in self.layers:
            attention = self.model.model.layers[layer_index].self_attn

            def hook(module, args, kwargs, layer_index=layer_index):
                hidden = args[0] if args else kwargs["hidden_states"]
                shape = (*hidden.shape[:-1], -1, module.head_dim)
                keys = module.k_norm(module.k_proj(hidden).view(shape)).transpose(1, 2)
                values = module.v_proj(hidden).view(shape).transpose(1, 2)
                self.states[layer_index] = (keys.detach(), values.detach())

            self.handles.append(
                attention.register_forward_pre_hook(hook, with_kwargs=True)
            )
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()


def question_prefix(tokenizer, question):
    text = f"QUESTION\n{question}\n\n"
    return tokenizer(text, add_special_tokens=False).input_ids


def wrong_question_mapping(cache, tokenizer):
    rows = []
    lengths = []
    for index in range(len(cache)):
        row = cache.load(index)["row"]
        rows.append(row)
        lengths.append(len(question_prefix(tokenizer, row["question"])))
    mapping = []
    for index in range(len(cache)):
        row = rows[index]
        candidates = []
        for other in range(len(cache)):
            if other == index:
                continue
            other_row = rows[other]
            if other_row.get("type") != row.get("type"):
                continue
            if other_row.get("answer_type") != row.get("answer_type"):
                continue
            candidates.append((abs(lengths[index] - lengths[other]), other))
        if not candidates:
            raise RuntimeError(f"No type-matched wrong question for {row['id']}")
        mapping.append(min(candidates)[1])
    return mapping


def hard_evidence_mapping(cache):
    rows = []
    lengths = []
    for index in range(len(cache)):
        payload = cache.load(index)
        rows.append(payload["row"])
        lengths.append(int(payload["keys"].shape[1]))
    mapping = []
    for index, row in enumerate(rows):
        current_titles = {
            normalize_answer(title) for title in row.get("supporting_titles", [])
        }
        current_bridge = normalize_answer(row.get("bridge_entity", ""))
        candidates = []
        for other, other_row in enumerate(rows):
            if other == index:
                continue
            if other_row.get("type") != row.get("type"):
                continue
            if other_row.get("answer_type") != row.get("answer_type"):
                continue
            if aliases_overlap(row["answer"], other_row["answer"]):
                continue
            other_titles = {
                normalize_answer(title)
                for title in other_row.get("supporting_titles", [])
            }
            if current_titles & other_titles:
                continue
            other_bridge = normalize_answer(other_row.get("bridge_entity", ""))
            if current_bridge and other_bridge and current_bridge == other_bridge:
                continue
            if normalize_answer(row["answer"]) in normalize_answer(
                evidence_block(other_row)
            ):
                continue
            length_gap = abs(lengths[index] - lengths[other])
            answer_gap = abs(
                len(str(row["answer"]).split())
                - len(str(other_row["answer"]).split())
            )
            candidates.append((length_gap, answer_gap, other))
        if not candidates:
            raise RuntimeError(f"No leakage-safe hard negative for {row['id']}")
        mapping.append(min(candidates)[2])
    return mapping


def capture_evidence(model, capture, prefix_ids, evidence_ids, layers, device, max_length):
    full_ids = list(prefix_ids) + list(evidence_ids)
    if len(full_ids) > max_length:
        raise RuntimeError(f"Sender input {len(full_ids)} exceeds max_length={max_length}")
    ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    capture.states.clear()
    model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    start, end = len(prefix_ids), len(full_ids)
    keys, values = [], []
    for layer_index in layers:
        key, value = capture.states[layer_index]
        keys.append(key[0, :, start:end, :].transpose(0, 1).float().cpu())
        values.append(value[0, :, start:end, :].transpose(0, 1).float().cpu())
    return torch.stack(keys).half(), torch.stack(values).half()


def state_metadata(source, prefix_length):
    metadata = dict(source["metadata"])
    metadata.update(
        {
            "prefix_token_count": int(prefix_length),
            "evidence_token_count": int(source["keys"].shape[1]),
            "question_tokens_transmitted": 0,
            "evidence_only_slice": True,
            "pre_rope_keys": True,
        }
    )
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-memory", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    cache = SenderNativeHeadwiseCache(args.base_memory, capacity=8)
    total = min(args.max_samples or len(cache), len(cache))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()
    model.requires_grad_(False)
    layers = list(cache.index["original_layer_indices"])
    wrong_questions = wrong_question_mapping(cache, tokenizer)
    hard_evidence = hard_evidence_mapping(cache)
    neutral_id = tokenizer(" neutral", add_special_tokens=False).input_ids[0]
    entries = []

    with NativeCapture(model, layers) as capture, torch.inference_mode():
        for index in tqdm(range(total), desc="p3e_l_cache_conditioned_native"):
            current = cache.load(index)
            wrong_q = cache.load(wrong_questions[index])
            wrong_e = cache.load(hard_evidence[index])
            q_prefix = question_prefix(tokenizer, current["row"]["question"])
            wrong_prefix = question_prefix(tokenizer, wrong_q["row"]["question"])
            prefixes = {
                "neutral_prefix": [neutral_id] * len(q_prefix),
                "wrong_question": wrong_prefix,
                "correct_question": q_prefix,
                "correct_question_hard_shuffled_evidence": q_prefix,
            }
            sources = {
                "neutral_prefix": current,
                "wrong_question": current,
                "correct_question": current,
                "correct_question_hard_shuffled_evidence": wrong_e,
            }
            states = {}
            for condition in CONDITIONS:
                source = sources[condition]
                evidence_ids = source["metadata"]["token_ids"]
                keys, values = capture_evidence(
                    model,
                    capture,
                    prefixes[condition],
                    evidence_ids,
                    layers,
                    device,
                    args.max_length,
                )
                if keys.shape[1] != len(evidence_ids):
                    raise RuntimeError("Evidence slice length changed")
                states[condition] = {
                    "keys": keys,
                    "values": values,
                    "metadata": state_metadata(source, len(prefixes[condition])),
                    "source_id": source["row"]["id"],
                    "source_answer": source["row"]["answer"],
                }
            filename = f"sample_{index:05d}.pt"
            torch.save(
                {
                    "row": current["row"],
                    "conditions": states,
                    "wrong_question_id": wrong_q["row"]["id"],
                    "hard_evidence_id": wrong_e["row"]["id"],
                },
                output / filename,
            )
            entries.append(
                {
                    "id": current["row"]["id"],
                    "file": filename,
                    "wrong_question_index": wrong_questions[index],
                    "hard_evidence_index": hard_evidence[index],
                }
            )

    result = {
        "status": "complete",
        "experiment": "P3-E-L conditioned Native KV cache",
        "samples": total,
        "entries": entries,
        "conditions": list(CONDITIONS),
        "shape": "[16,T,8,128]",
        "sender": args.model,
        "sender_config_sha256": file_sha256(Path(args.model) / "config.json"),
        "question_tokens_transmitted": 0,
        "receiver_sees_question_only": True,
    }
    write_json(output / "index.json", result)
    write_json(output / "SUCCESS.json", result)


if __name__ == "__main__":
    main()
