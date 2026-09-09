import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from p3d3_common import (
    answer_scores,
    hard_negative_mapping,
    load_receiver,
    normalize_answer,
    seed_everything,
)
from p3e_f_common import memory_to, sha256_file, write_json, write_jsonl
from p3e_h_common import load_c1_reader
from p3e_j_common import (
    PairedCanonicalTokenCache,
    exact_prompt_ids,
    load_decoder,
    manual_greedy_generate,
    manual_greedy_generate_ids,
    nearest_token_metrics,
    prompt_embeddings,
    reconstruction_metrics,
    template_token_ids,
    verify_tokenizer_alignment,
)


def summarize(records, condition):
    selected = [row for row in records if row["condition"] == condition]
    result = {
        "n": len(selected),
        "em": sum(row["em"] for row in selected) / len(selected),
        "f1": sum(row["f1"] for row in selected) / len(selected),
        "eos_rate": sum(row["output"]["eos_reached"] for row in selected) / len(selected),
        "average_output_tokens": sum(len(row["output"]["token_ids"]) for row in selected) / len(selected),
        "by_type": {},
    }
    for kind in ("bridge", "comparison"):
        group = [row for row in selected if row["type"] == kind]
        result["by_type"][kind] = {
            "n": len(group),
            "em": sum(row["em"] for row in group) / len(group),
            "f1": sum(row["f1"] for row in group) / len(group),
        }
    return result


@torch.inference_mode()
def decode_canonical(decoder, payload, device, permutation=None):
    keys = payload["keys"].to(device)
    values = payload["values"].to(device)
    mask = payload["mask"].to(device)
    if permutation is not None:
        keys = keys[:, permutation]
        values = values[:, permutation]
        mask = mask[permutation]
    return decoder(keys, values, mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--canonical-index", required=True)
    parser.add_argument("--native-index", required=True)
    parser.add_argument("--writer-checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--c1-reader", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cache = PairedCanonicalTokenCache(
        args.canonical_index, args.native_index, args.data, capacity=3
    )
    count = min(args.max_samples, len(cache))
    if count != 64:
        raise RuntimeError("P3-E-J requires the fixed validation64 split")
    if all("hard_negative_index" in entry for entry in cache.canonical_entries):
        negatives = [
            int(entry["hard_negative_index"]) for entry in cache.canonical_entries
        ]
        negative_source = "canonical_index"
    else:
        negatives = hard_negative_mapping(cache)
        negative_source = "deterministic_leakage_safe_reconstruction"
    model, tokenizer = load_receiver(args.model, device)
    decoder, decoder_checkpoint = load_decoder(args.decoder, device)
    decoder.requires_grad_(False)
    decoder.eval()
    c1_checkpoint = torch.load(args.c1_reader, map_location="cpu", weights_only=False)
    c1_reader = load_c1_reader(model, c1_checkpoint)
    conditions = [
        "question_only",
        "full_evidence_text",
        "exact_evidence_embeddings",
        "current_c1_headwise_reader",
        "canonical_soft_prefix",
        "sample_shuffled_canonical_soft_prefix",
        "token_order_shuffled_soft_prefix",
        "soft_prefix_off",
    ]
    records, pair_rows, prompt_rows = [], [], []
    reconstruction_rows = []
    nearest_predicted, nearest_ids, nearest_masks = [], [], []
    for index in tqdm(range(count), desc="p3e_j_eval64"):
        payload = cache.load(index)
        wrong = cache.load(negatives[index])
        row = payload["row"]
        verify_tokenizer_alignment(tokenizer, payload)
        evidence_ids = payload["token_ids"].to(device)
        exact_evidence = model.get_input_embeddings()(evidence_ids).float()
        soft = decode_canonical(decoder, payload, device)
        wrong_soft = decode_canonical(decoder, wrong, device)
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + index)
        permutation = torch.randperm(payload["keys"].shape[1], generator=generator, device=device)
        shuffled_tokens_soft = decode_canonical(
            decoder, payload, device, permutation=permutation
        )
        reconstruction_rows.append({
            "id": row["id"],
            **reconstruction_metrics(
                soft, exact_evidence, evidence_ids,
                payload["mask"].to(device),
            ),
        })
        remaining = 512 - sum(value.numel() for value in nearest_ids)
        if remaining > 0:
            positions = torch.nonzero(payload["mask"].to(device), as_tuple=False).flatten()[:remaining]
            nearest_predicted.append(soft[positions])
            nearest_ids.append(evidence_ids[positions])
            nearest_masks.append(torch.ones(positions.numel(), dtype=torch.bool, device=device))
        question_ids = exact_prompt_ids(tokenizer, row, [])
        full_ids = exact_prompt_ids(tokenizer, row, evidence_ids.tolist())
        predictions, token_outputs = {}, {}
        for condition in conditions:
            if condition in {"question_only", "soft_prefix_off"}:
                result = manual_greedy_generate_ids(
                    model, tokenizer, question_ids, args.max_new_tokens
                )
            elif condition == "full_evidence_text":
                result = manual_greedy_generate_ids(
                    model, tokenizer, full_ids, args.max_new_tokens
                )
            elif condition == "exact_evidence_embeddings":
                exact_prompt = prompt_embeddings(
                    model, tokenizer, row, exact_evidence
                )
                result = manual_greedy_generate(
                    model, tokenizer, exact_prompt, args.max_new_tokens
                )
            elif condition == "current_c1_headwise_reader":
                with c1_reader.inject(model, memory_to(payload, device)):
                    result = manual_greedy_generate_ids(
                        model, tokenizer, question_ids, args.max_new_tokens
                    )
            elif condition == "canonical_soft_prefix":
                result = manual_greedy_generate(
                    model, tokenizer,
                    prompt_embeddings(model, tokenizer, row, soft),
                    args.max_new_tokens,
                )
            elif condition == "sample_shuffled_canonical_soft_prefix":
                result = manual_greedy_generate(
                    model, tokenizer,
                    prompt_embeddings(model, tokenizer, row, wrong_soft),
                    args.max_new_tokens,
                )
            else:
                result = manual_greedy_generate(
                    model, tokenizer,
                    prompt_embeddings(model, tokenizer, row, shuffled_tokens_soft),
                    args.max_new_tokens,
                )
            em, f1 = answer_scores(result["prediction"], row["answer"])
            item = {
                "id": row["id"], "type": row["type"], "answer": row["answer"],
                "condition": condition, "em": em, "f1": f1, "output": result,
            }
            if condition == "sample_shuffled_canonical_soft_prefix":
                item.update({
                    "source_id": wrong["row"]["id"],
                    "source_answer": wrong["row"]["answer"],
                })
            records.append(item)
            predictions[condition] = result["prediction"]
            token_outputs[condition] = result["token_ids"]
        if token_outputs["full_evidence_text"] != token_outputs["exact_evidence_embeddings"]:
            raise RuntimeError("Exact Evidence inputs_embeds generation differs from input_ids")
        if token_outputs["question_only"] != token_outputs["soft_prefix_off"]:
            raise RuntimeError("soft_prefix_off does not exactly equal question_only")
        pair_rows.append({
            "id": row["id"],
            "correct_shuffled_prediction_switch": float(
                normalize_answer(predictions["canonical_soft_prefix"])
                != normalize_answer(predictions["sample_shuffled_canonical_soft_prefix"])
            ),
            "exact_embedding_generation_equal": True,
            "soft_prefix_off_equal": True,
        })
        prompt_rows.append({
            "id": row["id"],
            "question_only_ids": question_ids.tolist(),
            "full_evidence_ids": full_ids.tolist(),
            "evidence_token_ids": evidence_ids.tolist(),
            "template": "Evidence:\\n<body>\\n\\nQuestion:\\n<question>\\n\\nAnswer:\\nFINAL:",
        })
    write_jsonl(output / "per_sample_generation.jsonl", records)
    write_jsonl(output / "reconstruction_per_sample.jsonl", reconstruction_rows)
    write_jsonl(output / "prompts_and_token_ids.jsonl", prompt_rows)
    metrics = {condition: summarize(records, condition) for condition in conditions}
    nearest = nearest_token_metrics(
        torch.cat(nearest_predicted, dim=0),
        torch.cat(nearest_ids, dim=0),
        torch.cat(nearest_masks, dim=0),
        model.get_input_embeddings().weight,
        max_positions=512,
    )
    reconstruction = {
        "samples": len(reconstruction_rows),
        "tokens": sum(row["tokens"] for row in reconstruction_rows),
        "embedding_cosine": sum(
            row["embedding_cosine"] * row["tokens"] for row in reconstruction_rows
        ) / sum(row["tokens"] for row in reconstruction_rows),
        "embedding_mse": sum(
            row["embedding_mse"] * row["tokens"] for row in reconstruction_rows
        ) / sum(row["tokens"] for row in reconstruction_rows),
        "nearest_neighbor": {
            **nearest,
            "scope": "first_512_valid_validation_positions_in_fixed_sample_order",
        },
    }
    result = {
        "status": "complete",
        "experiment": "P3-E-J Canonical Soft-Prefix Executability Diagnosis",
        "diagnostic_only_not_final_communication_method": True,
        "samples": count,
        "conditions": metrics,
        "embedding_reconstruction": reconstruction,
        "correct_shuffled_f1_gap": (
            metrics["canonical_soft_prefix"]["f1"]
            - metrics["sample_shuffled_canonical_soft_prefix"]["f1"]
        ),
        "canonical_minus_full_text_f1": (
            metrics["canonical_soft_prefix"]["f1"]
            - metrics["full_evidence_text"]["f1"]
        ),
        "canonical_minus_current_c1_f1": (
            metrics["canonical_soft_prefix"]["f1"]
            - metrics["current_c1_headwise_reader"]["f1"]
        ),
        "prediction_switch_rate": sum(
            row["correct_shuffled_prediction_switch"] for row in pair_rows
        ) / count,
        "input_ids_exact_embeddings_generation_consistency": 1.0,
        "soft_prefix_off_question_only_consistency": 1.0,
        "hard_negative_mapping_source": negative_source,
        "decoder_metadata": decoder.metadata(),
        "decoder_checkpoint": args.decoder,
        "decoder_checkpoint_sha256": sha256_file(args.decoder),
        "fixed_c2_writer_checkpoint": args.writer_checkpoint,
        "fixed_c2_writer_sha256": sha256_file(args.writer_checkpoint),
        "c1_reader": args.c1_reader,
        "c1_reader_sha256": sha256_file(args.c1_reader),
        "manual_semantic_evaluation": "pending_blinded_CPW_review",
    }
    write_json(output / "SUCCESS.json", result)


if __name__ == "__main__":
    main()
