import argparse
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from p3d_common import answer_scores, compose_memory, load_receiver, memory_to, normalize_answer, permute_layers, permute_tokens, read_json, resize_memory, write_json, write_jsonl, zero_memory
from p3d2_common import GroundedEvidenceReader, canonical_cache, forward_grounded, generate_grounded, hard_negative_mapping
from train_p3d2_grounded_reader import TeacherCache, grounding_loss


def condition_memory(name, current, wrong, second_wrong, seed):
    tokens = current["keys"].shape[1]; wrong = resize_memory(wrong, tokens); second_wrong = resize_memory(second_wrong, tokens)
    if name == "correct": return current, True
    if name == "hard_shuffled": return wrong, True
    if name == "zero": return zero_memory(current), True
    if name == "reader_off": return current, False
    if name == "kv_mismatch": return compose_memory(wrong, second_wrong, tokens), True
    if name == "wrong_k_correct_v": return compose_memory(wrong, current, tokens), True
    if name == "correct_k_wrong_v": return compose_memory(current, wrong, tokens), True
    if name == "token_permutation": return permute_tokens(current, seed), True
    if name == "layer_permutation": return permute_layers(current), True
    raise ValueError(name)


def functional_diagnostics(trace, labels, teacher, student_logits):
    positions = (labels[0] != -100).nonzero(as_tuple=False).flatten(); count = min(len(positions), int(teacher["target_ids"].numel())); positions = positions[:count]
    target = teacher["teacher_hidden_delta"].to(student_logits.device).float(); q_rms = teacher["question_hidden_rms"].to(student_logits.device).float()
    cosines, ratios, target_ratios = [], [], []
    for layer, item in trace.items():
        delta = item["delta"][0].index_select(0, positions).float(); expected = target[layer, :count]
        cosines.append(F.cosine_similarity(delta, expected, dim=-1).mean())
        ratios.append((delta.square().mean(-1).sqrt() / q_rms[layer, :count].clamp_min(1e-5)).mean())
        target_ratios.append((expected.square().mean(-1).sqrt() / q_rms[layer, :count].clamp_min(1e-5)).mean())
    return {"residual_cosine": float(torch.stack(cosines).mean()), "reader_rms_ratio": float(torch.stack(ratios).mean()), "teacher_rms_ratio": float(torch.stack(target_ratios).mean()), "grounding_kl": float(grounding_loss(trace, positions, teacher["grounding_joint"]).detach())}


def aggregate(records):
    grouped = defaultdict(list)
    for row in records: grouped[row["condition"]].append(row)
    result = {}
    for condition, rows in grouped.items():
        item = {"n": len(rows), "em": sum(row["em"] for row in rows) / len(rows), "f1": sum(row["f1"] for row in rows) / len(rows), "source_em": sum(row["source_em"] for row in rows) / len(rows), "insufficient_rate": sum(row["is_insufficient"] for row in rows) / len(rows), "eos_rate": sum(row["eos_reached"] for row in rows) / len(rows), "compatibility_score": sum(row["compatibility_score"] for row in rows) / len(rows)}
        for kind in ("bridge", "comparison"):
            subset = [row for row in rows if row["question_type"] == kind]
            if subset: item[f"{kind}_f1"] = sum(row["f1"] for row in subset) / len(subset)
        result[condition] = item
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--model", required=True); parser.add_argument("--protocol", required=True)
    parser.add_argument("--teacher", required=True); parser.add_argument("--checkpoint", required=True); parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--out", required=True); parser.add_argument("--max-samples", type=int, default=0); parser.add_argument("--max-new-tokens", type=int, default=32); parser.add_argument("--seed", type=int, default=1234); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); device = torch.device(args.device); protocol = read_json(args.protocol); cache = canonical_cache(protocol, args.split); teachers = TeacherCache(args.teacher)
    model, tokenizer = load_receiver(args.model, device); checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False); metadata = checkpoint["reader_metadata"]
    reader = GroundedEvidenceReader(model, metadata["groups"], metadata["memory_dim"], metadata["active_layers"], metadata["shared_blocks"], metadata["rank"], metadata["adapter_rank"]).to(device)
    if reader.metadata() != metadata: raise RuntimeError("Reader checkpoint interface mismatch")
    reader.load_state_dict(checkpoint["reader"]); reader.eval(); negative = hard_negative_mapping(cache); limit = min(args.max_samples or len(cache), len(cache))
    conditions = ("correct", "hard_shuffled", "zero", "reader_off", "kv_mismatch", "wrong_k_correct_v", "correct_k_wrong_v", "token_permutation", "layer_permutation")
    records, diagnostics = [], []
    for index in tqdm(range(limit), desc=f"p3d2_eval_{checkpoint['configuration']}"):
        payload = cache.load(index); wrong_index = negative[index]; second_index = negative[wrong_index]
        wrong_payload, second_payload = cache.load(wrong_index), cache.load(second_index); teacher = teachers.load(index)
        current, wrong, second_wrong = memory_to(payload, device), memory_to(wrong_payload, device), memory_to(second_payload, device)
        q_hidden = teacher["question_prompt_hidden"].to(device).float()
        for condition in conditions:
            memory, enabled = condition_memory(condition, current, wrong, second_wrong, args.seed + index)
            result = generate_grounded(model, tokenizer, reader, payload["row"], memory, args.max_new_tokens, enabled)
            em, f1 = answer_scores(result["prediction"], payload["row"]["answer"]); source_em, source_f1 = answer_scores(result["prediction"], wrong_payload["row"]["answer"])
            compatibility = float(reader.compatibility_from_hidden(q_hidden, memory)) if enabled else 0.0
            records.append({"sample_index": index, "sample_id": payload["row"].get("id", str(index)), "condition": condition, "question_type": payload["row"].get("type", "unknown"), "question": payload["row"]["question"], "gold_answer": payload["row"]["answer"], "source_answer": wrong_payload["row"]["answer"], "prediction": result["prediction"], "raw_generation": result["text"], "parse_status": result["parse_status"], "em": em, "f1": f1, "source_em": source_em, "source_f1": source_f1, "is_insufficient": float(normalize_answer(result["prediction"]) == "insufficient"), "eos_reached": float(result["eos_reached"]), "compatibility_score": compatibility, "reader_diagnostics": result["diagnostics"] if condition == "correct" else []})
        trace = {}; _, logits, labels = forward_grounded(model, tokenizer, reader, payload["row"], current, payload["row"]["answer"], 1024, device, True, trace)
        diagnostics.append({"sample_index": index, **functional_diagnostics(trace, labels, teacher, logits)})
    output = Path(args.out); output.mkdir(parents=True, exist_ok=True); write_jsonl(output / "per_sample_generation.jsonl", records); write_jsonl(output / "functional_diagnostics.jsonl", diagnostics)
    conditions_summary = aggregate(records); correct, shuffled = conditions_summary["correct"], conditions_summary["hard_shuffled"]
    write_json(output / "SUCCESS.json", {"status": "complete", "configuration": checkpoint["configuration"], "samples": limit, "conditions": conditions_summary, "correct_minus_shuffled_f1": correct["f1"] - shuffled["f1"], "functional_diagnostics": {name: sum(row[name] for row in diagnostics) / max(1, len(diagnostics)) for name in ("residual_cosine", "reader_rms_ratio", "teacher_rms_ratio", "grounding_kl")}, "reader_parameters": sum(parameter.numel() for parameter in reader.parameters()), "receiver_parameters_updated": 0, "writer_parameters_updated": 0, "checkpoint": args.checkpoint})


if __name__ == "__main__": main()
