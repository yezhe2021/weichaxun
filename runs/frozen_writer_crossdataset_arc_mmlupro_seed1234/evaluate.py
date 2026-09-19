import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from common import ROOT, final_logits, load_model, log, read_json
from translators import (ResidualAdapter, gemma_to_qwen, llama_to_qwen,
                         qwen_to_gemma, qwen_to_llama)


def load_module(module, path, kinds=()):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if kinds and payload.get("kind") not in kinds:
        raise RuntimeError(f"Unexpected checkpoint kind {payload.get('kind')} for {path}")
    module.load_state_dict(payload["state"], strict=True)
    return module.cuda().eval().requires_grad_(False)


def translate(module, key, value):
    with torch.amp.autocast("cuda", dtype=torch.float16):
        output = module(key.cuda()[None], value.cuda()[None])
    return output[0][0], output[1][0]


def adapt(module, key, value):
    with torch.amp.autocast("cuda", dtype=torch.float16):
        output = module(key[None], value[None])
    return output[0][0], output[1][0]


def choice_kl(student, teacher, temperature):
    first = F.log_softmax(student.float() / temperature, dim=-1)
    second = F.log_softmax(teacher.float() / temperature, dim=-1)
    return (F.kl_div(first, second, reduction="sum", log_target=True) * temperature ** 2).item()


def representation(prediction, target):
    prediction = prediction.float()
    target = target.to(prediction.device).float()
    return {"nmse": ((prediction-target).square().mean() / target.square().mean().clamp_min(1e-8)).item(),
            "cosine": F.cosine_similarity(prediction, target, dim=-1).mean().item()}


def evaluate_state(model, fields, receiver, key, value, oracle_logits, oracle_key=None, oracle_value=None):
    full_key = torch.cat((receiver["question_k"].cuda(), key.cuda()), dim=1)
    full_value = torch.cat((receiver["question_v"].cuda(), value.cuda()), dim=1)
    indices = torch.tensor(fields["choice_ids"], device="cuda")
    logits = final_logits(model, fields["receiver_answer"], full_key, full_value)[indices]
    prediction = int(logits.argmax())
    result = {"prediction": prediction, "choice_logits": logits.cpu().tolist(),
              "choice_kl_to_oracle": choice_kl(logits, oracle_logits, 1.0),
              "oracle_agreement": prediction == int(oracle_logits.argmax())}
    if oracle_key is not None:
        result["k_representation"] = representation(key, oracle_key)
        result["v_representation"] = representation(value, oracle_value)
    return result


def oracle_state(model, fields, receiver, key, value):
    full_key = torch.cat((receiver["question_k"].cuda(), key.cuda()), dim=1)
    full_value = torch.cat((receiver["question_v"].cuda(), value.cuda()), dim=1)
    indices = torch.tensor(fields["choice_ids"], device="cuda")
    return final_logits(model, fields["receiver_answer"], full_key, full_value)[indices]


def no_memory(model, fields, receiver):
    indices = torch.tensor(fields["choice_ids"], device="cuda")
    logits = final_logits(model, fields["receiver_answer"],
                          receiver["question_k"].cuda(), receiver["question_v"].cuda())[indices]
    return logits


def base_record(row, receiver, full_logits, no_logits):
    return {"id": row["id"], "category": row["category"], "gold_index": row["gold_index"],
            "conditions": {
                "full_native": {"prediction": int(full_logits.argmax()),
                                "choice_logits": full_logits.tolist()},
                "no_memory": {"prediction": int(no_logits.argmax()),
                              "choice_logits": no_logits.cpu().tolist()}}}


def load_rows(mode, dataset):
    return read_json(ROOT / "runs" / mode / dataset / "manifest.json")["rows"]


def load_cache(mode, dataset, sample_id):
    return torch.load(ROOT / "runs" / mode / dataset / "cache/routes" / f"{sample_id}.pt",
                      map_location="cpu", weights_only=True)


@torch.no_grad()
def evaluate_qwen(cfg, mode, dataset):
    cp = cfg["checkpoints"]
    lq = load_module(llama_to_qwen(), cp["llama_to_qwen_stage_a"])
    lq_res = load_module(ResidualAdapter(36, 8, 128), cp["llama_to_qwen_residual"], ("residual",))
    gq = load_module(gemma_to_qwen(), cp["gemma_to_qwen_stage_a"])
    gq_res = load_module(ResidualAdapter(36, 8, 128), cp["gemma_to_qwen_residual"], ("residual",))
    model = load_model(cfg, "qwen"); records = []
    try:
        rows = load_rows(mode, dataset)
        for number, row in enumerate(rows, 1):
            cache = load_cache(mode, dataset, row["id"]); receiver = cache["receiver"]["qwen"]
            fields = row["encoded"]["qwen"]
            no_logits = no_memory(model, fields, receiver)
            record = base_record(row, receiver, receiver["full_choice_logits"], no_logits)
            for prefix, router, base, residual in (
                    ("lq", "llama", lq, lq_res), ("gq", "gemma", gq, gq_res)):
                source = cache["routes"][router][router]
                oracle = cache["routes"][router]["qwen"]
                oracle_logits = oracle_state(model, fields, receiver, oracle["key"], oracle["value"])
                stage_k, stage_v = translate(base, source["key"], source["value"])
                residual_k, residual_v = adapt(residual, stage_k, stage_v)
                record["conditions"][f"{prefix}_oracle32"] = {
                    "prediction": int(oracle_logits.argmax()), "choice_logits": oracle_logits.cpu().tolist(),
                    "choice_kl_to_oracle": 0.0, "oracle_agreement": True}
                record["conditions"][f"{prefix}_stage_a"] = evaluate_state(
                    model, fields, receiver, stage_k, stage_v, oracle_logits, oracle["key"], oracle["value"])
                record["conditions"][f"{prefix}_residual"] = evaluate_state(
                    model, fields, receiver, residual_k, residual_v, oracle_logits, oracle["key"], oracle["value"])
            records.append(record)
            if number % 16 == 0 or number == len(rows): log(f"{dataset} qwen receiver: {number}/{len(rows)}")
    finally:
        del model, lq, lq_res, gq, gq_res; gc.collect(); torch.cuda.empty_cache()
    return records


@torch.no_grad()
def evaluate_gemma(cfg, mode, dataset):
    cp = cfg["checkpoints"]
    lq = load_module(llama_to_qwen(), cp["llama_to_qwen_stage_a"])
    lq_res = load_module(ResidualAdapter(36, 8, 128), cp["llama_to_qwen_residual"], ("residual",))
    qg = load_module(qwen_to_gemma(), cp["qwen_to_gemma_stage_a"])
    old = load_module(ResidualAdapter(34, 4, 256), cp["qwen_to_gemma_old_residual"], ("residual",))
    mixed = load_module(ResidualAdapter(34, 4, 256), cp["qwen_to_gemma_mixed_residual"],
                        ("mixed_receiver_residual", "residual"))
    model = load_model(cfg, "gemma"); records = []
    try:
        rows = load_rows(mode, dataset)
        for number, row in enumerate(rows, 1):
            cache = load_cache(mode, dataset, row["id"]); receiver = cache["receiver"]["gemma"]
            fields = row["encoded"]["gemma"]
            record = base_record(row, receiver, receiver["full_choice_logits"], no_memory(model, fields, receiver))
            # Natural one-hop Qwen router.
            qsource = cache["routes"]["qwen"]["qwen"]
            qoracle = cache["routes"]["qwen"]["gemma"]
            qoracle_logits = oracle_state(model, fields, receiver, qoracle["key"], qoracle["value"])
            qa_k, qa_v = translate(qg, qsource["key"], qsource["value"])
            qo_k, qo_v = adapt(old, qa_k, qa_v); qm_k, qm_v = adapt(mixed, qa_k, qa_v)
            record["conditions"]["qg_oracle32"] = {"prediction": int(qoracle_logits.argmax()),
                "choice_logits": qoracle_logits.cpu().tolist(), "choice_kl_to_oracle": 0.0, "oracle_agreement": True}
            for name, key, value in (("qg_stage_a", qa_k, qa_v), ("qg_old_residual", qo_k, qo_v),
                                     ("qg_mixed_residual", qm_k, qm_v)):
                record["conditions"][name] = evaluate_state(
                    model, fields, receiver, key, value, qoracle_logits, qoracle["key"], qoracle["value"])

            # Llama router, canonical and diagnostic two-hop states.
            source = cache["routes"]["llama"]["llama"]
            oracle = cache["routes"]["llama"]["gemma"]
            oracle_logits = oracle_state(model, fields, receiver, oracle["key"], oracle["value"])
            q_a_k, q_a_v = translate(lq, source["key"], source["value"])
            q_b_k, q_b_v = adapt(lq_res, q_a_k, q_a_v)
            g_aa_k, g_aa_v = translate(qg, q_a_k, q_a_v)
            g_ba_k, g_ba_v = translate(qg, q_b_k, q_b_v)
            g_aa_old_k, g_aa_old_v = adapt(old, g_aa_k, g_aa_v)
            g_aa_mix_k, g_aa_mix_v = adapt(mixed, g_aa_k, g_aa_v)
            g_ba_mix_k, g_ba_mix_v = adapt(mixed, g_ba_k, g_ba_v)
            record["conditions"]["lqg_oracle32"] = {"prediction": int(oracle_logits.argmax()),
                "choice_logits": oracle_logits.cpu().tolist(), "choice_kl_to_oracle": 0.0, "oracle_agreement": True}
            for name, key, value in (
                    ("lqg_stage_a_stage_a", g_aa_k, g_aa_v),
                    ("lqg_qwen_residual_stage_a", g_ba_k, g_ba_v),
                    ("lqg_stage_a_old_residual", g_aa_old_k, g_aa_old_v),
                    ("lqg_stage_a_mixed_residual", g_aa_mix_k, g_aa_mix_v),
                    ("lqg_both_residuals", g_ba_mix_k, g_ba_mix_v)):
                record["conditions"][name] = evaluate_state(
                    model, fields, receiver, key, value, oracle_logits, oracle["key"], oracle["value"])
            records.append(record)
            if number % 16 == 0 or number == len(rows): log(f"{dataset} gemma receiver: {number}/{len(rows)}")
    finally:
        del model, lq, lq_res, qg, old, mixed; gc.collect(); torch.cuda.empty_cache()
    return records


@torch.no_grad()
def evaluate_llama(cfg, mode, dataset):
    cp = cfg["checkpoints"]
    gq = load_module(gemma_to_qwen(), cp["gemma_to_qwen_stage_a"])
    gq_res = load_module(ResidualAdapter(36, 8, 128), cp["gemma_to_qwen_residual"], ("residual",))
    ql = load_module(qwen_to_llama(), cp["qwen_to_llama_stage_a"])
    ql_res = load_module(ResidualAdapter(28, 8, 128), cp["qwen_to_llama_residual"], ("residual",))
    model = load_model(cfg, "llama"); records = []
    try:
        rows = load_rows(mode, dataset)
        for number, row in enumerate(rows, 1):
            cache = load_cache(mode, dataset, row["id"]); receiver = cache["receiver"]["llama"]
            fields = row["encoded"]["llama"]
            record = base_record(row, receiver, receiver["full_choice_logits"], no_memory(model, fields, receiver))
            # Natural one-hop Qwen router.
            qsource = cache["routes"]["qwen"]["qwen"]
            qoracle = cache["routes"]["qwen"]["llama"]
            qoracle_logits = oracle_state(model, fields, receiver, qoracle["key"], qoracle["value"])
            qa_k, qa_v = translate(ql, qsource["key"], qsource["value"])
            qb_k, qb_v = adapt(ql_res, qa_k, qa_v)
            record["conditions"]["ql_oracle32"] = {"prediction": int(qoracle_logits.argmax()),
                "choice_logits": qoracle_logits.cpu().tolist(), "choice_kl_to_oracle": 0.0, "oracle_agreement": True}
            record["conditions"]["ql_stage_a"] = evaluate_state(
                model, fields, receiver, qa_k, qa_v, qoracle_logits, qoracle["key"], qoracle["value"])
            record["conditions"]["ql_residual"] = evaluate_state(
                model, fields, receiver, qb_k, qb_v, qoracle_logits, qoracle["key"], qoracle["value"])

            # Gemma router reverse two-hop.
            source = cache["routes"]["gemma"]["gemma"]
            oracle = cache["routes"]["gemma"]["llama"]
            oracle_logits = oracle_state(model, fields, receiver, oracle["key"], oracle["value"])
            q_a_k, q_a_v = translate(gq, source["key"], source["value"])
            q_b_k, q_b_v = adapt(gq_res, q_a_k, q_a_v)
            l_aa_k, l_aa_v = translate(ql, q_a_k, q_a_v)
            l_ba_k, l_ba_v = translate(ql, q_b_k, q_b_v)
            l_aa_res_k, l_aa_res_v = adapt(ql_res, l_aa_k, l_aa_v)
            l_ba_res_k, l_ba_res_v = adapt(ql_res, l_ba_k, l_ba_v)
            record["conditions"]["gql_oracle32"] = {"prediction": int(oracle_logits.argmax()),
                "choice_logits": oracle_logits.cpu().tolist(), "choice_kl_to_oracle": 0.0, "oracle_agreement": True}
            for name, key, value in (
                    ("gql_stage_a_stage_a", l_aa_k, l_aa_v),
                    ("gql_qwen_residual_stage_a", l_ba_k, l_ba_v),
                    ("gql_stage_a_llama_residual", l_aa_res_k, l_aa_res_v),
                    ("gql_both_residuals", l_ba_res_k, l_ba_res_v)):
                record["conditions"][name] = evaluate_state(
                    model, fields, receiver, key, value, oracle_logits, oracle["key"], oracle["value"])
            records.append(record)
            if number % 16 == 0 or number == len(rows): log(f"{dataset} llama receiver: {number}/{len(rows)}")
    finally:
        del model, gq, gq_res, ql, ql_res; gc.collect(); torch.cuda.empty_cache()
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--mode", choices=("smoke", "study"), default="study")
    parser.add_argument("--dataset", choices=("arc_challenge", "mmlu_pro"), required=True)
    parser.add_argument("--receiver", choices=("qwen", "gemma", "llama"), required=True)
    args = parser.parse_args(); cfg = read_json(args.config)
    torch.manual_seed(cfg["seed"])
    records = {"qwen": evaluate_qwen, "gemma": evaluate_gemma, "llama": evaluate_llama}[
        args.receiver](cfg, args.mode, args.dataset)
    output = ROOT / "runs" / args.mode / args.dataset / "results" / f"{args.receiver}_receiver.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in records: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    log(f"DONE {args.dataset} {args.receiver} receiver -> {output}")


if __name__ == "__main__":
    main()
