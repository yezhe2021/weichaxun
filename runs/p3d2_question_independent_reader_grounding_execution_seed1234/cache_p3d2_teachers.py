import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from p3d_common import full_text_prompt, load_receiver, question_prompt, read_json, write_json
from p3d2_common import canonical_cache, load_span_probe, span_teacher


def run_path(model, tokenizer, prompt, answer, max_length, device):
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    suffix_ids = tokenizer(" " + answer + (tokenizer.eos_token or ""), add_special_tokens=False).input_ids
    if len(prompt_ids) + len(suffix_ids) > max_length:
        raise RuntimeError(f"Teacher sequence length {len(prompt_ids) + len(suffix_ids)} exceeds {max_length}; increase --max-length")
    ids = torch.tensor([prompt_ids + suffix_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids); labels = ids.clone(); labels[:, :len(prompt_ids)] = -100
    output = model(input_ids=ids, attention_mask=mask, use_cache=False, output_hidden_states=True, return_dict=True)
    label_positions = (labels[0] != -100).nonzero(as_tuple=False).flatten()
    shifted_labels = labels[:, 1:]; target_ids = shifted_labels[shifted_labels != -100]
    logits = output.logits[:, :-1, :][shifted_labels != -100]
    if len(label_positions) != len(target_ids): raise RuntimeError("Answer hidden/logit alignment failed")
    return output, label_positions, target_ids, logits


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True); parser.add_argument("--protocol", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--max-length", type=int, default=1024); parser.add_argument("--max-samples", type=int, default=0); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); device = torch.device(args.device); protocol = read_json(args.protocol)
    cache = canonical_cache(protocol, args.split); model, tokenizer = load_receiver(args.model, device); probe, probe_path = load_span_probe(protocol, device)
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True); entries = []
    limit = min(args.max_samples or len(cache), len(cache))
    for index in tqdm(range(limit), desc=f"p3d2_teacher_{args.split}"):
        payload = cache.load(index); row = payload["row"]; answer = row["answer"]
        q_output, q_positions, q_targets, q_logits = run_path(model, tokenizer, question_prompt(tokenizer, row), answer, args.max_length, device)
        f_output, f_positions, f_targets, f_logits = run_path(model, tokenizer, full_text_prompt(tokenizer, row), answer, args.max_length, device)
        count = min(len(q_targets), len(f_targets))
        if count == 0 or not torch.equal(q_targets[:count], f_targets[:count]): raise RuntimeError(f"Teacher target mismatch at {index}")
        q_positions, f_positions, targets = q_positions[:count], f_positions[:count], q_targets[:count]
        layer_deltas, q_rms, q_prompt_hidden = [], [], []
        prompt_end = max(0, int(q_positions[0]) - 1)
        for layer in range(int(model.config.num_hidden_layers)):
            q_answer = q_output.hidden_states[layer + 1][0].index_select(0, q_positions).float()
            f_answer = f_output.hidden_states[layer + 1][0].index_select(0, f_positions).float()
            layer_deltas.append((f_answer - q_answer).half().cpu())
            q_rms.append(q_answer.square().mean(dim=-1).sqrt().half().cpu())
            q_prompt_hidden.append(q_output.hidden_states[layer][0, prompt_end].half().cpu())
        q_gold = q_logits[:count].gather(-1, targets[:, None]).squeeze(-1)
        f_gold = f_logits[:count].gather(-1, targets[:, None]).squeeze(-1)
        grounding = span_teacher(probe, payload, device)
        record = {
            "row_id": row.get("id", str(index)), "target_ids": targets.cpu(),
            "teacher_hidden_delta": torch.stack(layer_deltas), "question_hidden_rms": torch.stack(q_rms),
            "question_prompt_hidden": torch.stack(q_prompt_hidden), "question_gold_logits": q_gold.half().cpu(),
            "teacher_gold_logit_delta": (f_gold - q_gold).half().cpu(),
            "grounding_joint": grounding["joint"].half().cpu(), "grounding_layer_weights": grounding["layer_weights"].half().cpu(),
        }
        filename = f"sample_{index:05d}.pt"; torch.save(record, output / filename)
        entries.append({"index": index, "id": record["row_id"], "file": filename, "answer_tokens": count})
    write_json(output / "index.json", {"status": "complete", "split": args.split, "entries": entries, "span_probe": str(probe_path), "receiver_model": args.model, "max_length": args.max_length})
    write_json(output / "SUCCESS.json", {"status": "complete", "split": args.split, "samples": len(entries), "teacher_paths": ["question_only", "question_plus_gold_evidence"], "answer_relative_alignment": True, "span_probe_frozen": True})


if __name__ == "__main__": main()
