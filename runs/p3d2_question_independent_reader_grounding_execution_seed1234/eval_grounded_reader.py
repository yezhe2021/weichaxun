import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from p3d2_common import (
    SharedGroundedReader, TeacherTraceCache, aggregate_scores, answer_scores, build_cache,
    compose_memory, extract_prediction, hard_negative_mapping, load_receiver, load_span_probe,
    memory_from_payload, normalize_answer, permute_layers, permute_tokens, question_prompt,
    prediction_position_mask, read_json, resize_memory, seed_everything, span_teacher, write_json, write_jsonl,
    zero_memory,
)
from train_grounded_reader import execution_losses, grounding_loss, reader_forward


@torch.inference_mode()
def generate(model, tokenizer, reader, row, memory, enabled, max_new_tokens):
    encoded = tokenizer(question_prompt(tokenizer, row), return_tensors="pt", add_special_tokens=False)
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    kwargs = {
        **encoded,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    trace = {}
    if enabled:
        with reader.inject(model, memory, trace):
            output = model.generate(**kwargs)
    else:
        output = model.generate(**kwargs)
    tokens = output[0, encoded["input_ids"].shape[1]:]
    scores = [values["compatibility_score"].float().mean().item() for values in trace.values()]
    return tokenizer.decode(tokens, skip_special_tokens=True), len(tokens), float(np.mean(scores)) if scores else 0.0


def make_conditions(index, cache, negative, device, seed):
    payload = cache.load(index)
    correct = memory_from_payload(payload, device)
    wrong_index = negative[index]
    wrong_payload = cache.load(wrong_index)
    wrong = resize_memory(memory_from_payload(wrong_payload, device), correct["keys"].shape[1])
    second_index = negative[wrong_index]
    if second_index in {index, wrong_index}:
        second_index = (wrong_index + 1) % len(cache)
        if second_index == index:
            second_index = (second_index + 1) % len(cache)
    second_payload = cache.load(second_index)
    second = resize_memory(memory_from_payload(second_payload, device), correct["keys"].shape[1])
    return payload, wrong_payload, {
        "correct": (correct, True, None),
        "hard_shuffled": (wrong, True, wrong_payload),
        "zero": (zero_memory(correct), True, None),
        "reader_off": (correct, False, None),
        "kv_mismatch": (compose_memory(wrong, second), True, wrong_payload),
        "wrong_k_correct_v": (compose_memory(wrong, correct), True, wrong_payload),
        "correct_k_wrong_v": (compose_memory(correct, wrong), True, wrong_payload),
        "token_permutation": (permute_tokens(correct, seed + index), True, None),
        "layer_permutation": (permute_layers(correct), True, None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--span-probe", required=True)
    parser.add_argument("--teacher-cache", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    protocol = read_json(args.protocol)
    cache = build_cache(protocol, args.split)
    teacher_cache = TeacherTraceCache(args.teacher_cache)
    if len(cache) != len(teacher_cache.entries):
        raise RuntimeError("Execution-teacher and Canonical cache lengths differ")
    negative = hard_negative_mapping(cache)
    model, tokenizer = load_receiver(args.model, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = checkpoint["reader_metadata"]
    reader = SharedGroundedReader(
        model,
        metadata["active_layers"],
        groups=metadata["groups"],
        memory_dim=metadata["memory_dim"],
        rank=metadata["rank"],
        adapter_rank=metadata["adapter_rank"],
        compatibility_rank=metadata["compatibility_rank"],
    ).to(device)
    if reader.metadata() != metadata:
        raise RuntimeError("Reader checkpoint interface mismatch")
    reader.load_state_dict(checkpoint["reader"])
    reader.eval()
    probe = load_span_probe(args.span_probe, device)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Qwen3-4B backbone is not frozen")

    limit = min(len(cache), args.max_samples or len(cache))
    records, diagnostic_rows = [], []
    generated_by_sample = {}
    for index in tqdm(range(limit), desc=f"p3d2_eval_{metadata['active_layers']}"):
        payload, wrong_payload, conditions = make_conditions(index, cache, negative, device, args.seed)
        row = payload["row"]
        generated_by_sample[index] = {}
        compatibility_by_condition = {}
        for condition, (memory, enabled, source_payload) in conditions.items():
            text, generated_length, compatibility = generate(
                model, tokenizer, reader, row, memory, enabled, args.max_new_tokens,
            )
            prediction, _ = extract_prediction(text)
            em, f1 = answer_scores(prediction, row["answer"])
            source_hit = 0.0
            if source_payload is not None:
                source_answer = normalize_answer(source_payload["row"]["answer"])
                source_hit = float(bool(source_answer) and source_answer in normalize_answer(prediction))
            records.append({
                "id": row["id"],
                "condition": condition,
                "question_type": row.get("type", "unknown"),
                "prediction": prediction,
                "gold": row["answer"],
                "em": em,
                "f1": f1,
                "generated_length": generated_length,
                "compatibility_score": compatibility,
                "wrong_memory_source_answer_hit": source_hit,
            })
            generated_by_sample[index][condition] = prediction
            compatibility_by_condition[condition] = compatibility

        correct_memory = conditions["correct"][0]
        output, labels, trace = reader_forward(
            model, tokenizer, reader, row, correct_memory, args.max_length, device,
        )
        probe_output = span_teacher(probe, payload, device)
        ground, ground_trace = grounding_loss(
            trace, prediction_position_mask(labels), probe_output, correct_memory, metadata["active_layers"],
        )
        execution, norm, execution_metrics = execution_losses(
            output, labels, trace, teacher_cache.load(index), metadata["active_layers"], device,
        )
        correct_score = compatibility_by_condition["correct"]
        wrong_score = compatibility_by_condition["hard_shuffled"]
        diagnostic_rows.append({
            "id": row["id"],
            "grounding_loss": float(ground),
            "execution_loss": float(execution),
            "norm_loss": float(norm),
            "execution_cosine_similarity": 1.0 - float(execution_metrics["cosine"]),
            "execution_normalized_mse": float(execution_metrics["normalized_mse"]),
            "gold_logit_delta_loss": float(execution_metrics["logit_delta"]),
            "reader_residual_rms_ratio": float(execution_metrics["reader_ratio"]),
            "teacher_residual_rms_ratio": float(execution_metrics["teacher_ratio"]),
            "compatibility_correct": float(correct_score > wrong_score),
            "zero_equals_reader_off": float(
                generated_by_sample[index]["zero"] == generated_by_sample[index]["reader_off"]
            ),
            "span_attention_mass": float(
                (ground_trace["student_router"][:, None] * ground_trace["student_attention"])
                .sum(0)[correct_memory["answer_token_mask"] | correct_memory["support_token_mask"]]
                .sum()
            ),
        })

    diagnostics_by_id = {row["id"]: row for row in diagnostic_rows}
    for row in records:
        diagnostic = diagnostics_by_id[row["id"]]
        row["compatibility_correct"] = diagnostic["compatibility_correct"]
    summary = aggregate_scores(records)
    for condition in summary:
        if not isinstance(summary[condition], dict):
            continue
        subset = [row for row in records if row["condition"] == condition]
        summary[condition]["generated_length"] = float(np.mean([row["generated_length"] for row in subset]))
        summary[condition]["wrong_memory_source_answer_hit"] = float(
            np.mean([row["wrong_memory_source_answer_hit"] for row in subset])
        )
    summary["diagnostics"] = {
        key: float(np.mean([row[key] for row in diagnostic_rows]))
        for key in diagnostic_rows[0]
        if key != "id"
    }
    summary["checkpoint"] = args.checkpoint
    summary["layer_config"] = checkpoint["layer_config"]
    summary["reader_parameters"] = sum(parameter.numel() for parameter in reader.parameters())
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "predictions.jsonl", records)
    write_jsonl(output_dir / "diagnostics.jsonl", diagnostic_rows)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "SUCCESS.json", {
        "status": "complete",
        "split": args.split,
        "samples": limit,
        "conditions": list(conditions),
        "receiver_parameters_updated": 0,
        "writer_parameters_updated": 0,
    })


if __name__ == "__main__":
    main()
