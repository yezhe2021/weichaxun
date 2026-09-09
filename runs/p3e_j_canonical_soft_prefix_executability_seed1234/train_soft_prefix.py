import argparse
import random
from pathlib import Path

import torch
from tqdm import tqdm

from p3d3_common import load_receiver, seed_everything
from p3e_f_common import sha256_file, write_json, write_jsonl
from p3e_j_common import (
    PairedCanonicalTokenCache,
    SoftPrefixDecoder,
    answer_kl,
    answer_mean_nll,
    assert_frozen_gradients,
    assert_optimizer_only_decoder,
    build_teacher_forcing_batch,
    decoder_checkpoint,
    embedding_reconstruction_loss,
    evidence_embedding_rms,
    hidden_alignment_loss,
    load_decoder,
    reconstruction_metrics,
    verify_tokenizer_alignment,
)


@torch.inference_mode()
def exact_embedding_interface_check(model, tokenizer, payload, device):
    token_ids = payload["token_ids"].to(device)
    target = model.get_input_embeddings()(token_ids)
    batch = build_teacher_forcing_batch(
        model, tokenizer, payload["row"], token_ids, target, device
    )
    kwargs = {
        "attention_mask": batch["attention_mask"],
        "position_ids": batch["position_ids"],
        "use_cache": True,
        "return_dict": True,
    }
    by_ids = model(input_ids=batch["ids"], **kwargs)
    by_embeds = model(inputs_embeds=batch["teacher_embeddings"], **kwargs)
    prefill_difference = (by_ids.logits.float() - by_embeds.logits.float()).abs()
    next_token = by_ids.logits[:, -1].argmax(dim=-1, keepdim=True)
    next_mask = torch.ones(
        1, batch["ids"].shape[1] + 1, dtype=torch.long, device=device
    )
    next_position = torch.tensor(
        [[batch["ids"].shape[1]]], dtype=torch.long, device=device
    )
    next_ids = model(
        input_ids=next_token,
        attention_mask=next_mask,
        position_ids=next_position,
        past_key_values=by_ids.past_key_values,
        use_cache=True,
        return_dict=True,
    )
    next_embeds = model(
        input_ids=next_token,
        attention_mask=next_mask,
        position_ids=next_position,
        past_key_values=by_embeds.past_key_values,
        use_cache=True,
        return_dict=True,
    )
    cache_difference = (next_ids.logits.float() - next_embeds.logits.float()).abs()
    result = {
        "prefill_max_abs_logit_difference": float(prefill_difference.max()),
        "prefill_mean_abs_logit_difference": float(prefill_difference.mean()),
        "cache_next_max_abs_logit_difference": float(cache_difference.max()),
        "cache_next_mean_abs_logit_difference": float(cache_difference.mean()),
        "next_token_equal": bool(
            next_ids.logits[:, -1].argmax(dim=-1).item()
            == next_embeds.logits[:, -1].argmax(dim=-1).item()
        ),
        "position_ids_equal": True,
        "causal_mask_equal": True,
        "sequence_tokens": int(batch["ids"].shape[1]),
    }
    if result["prefill_max_abs_logit_difference"] > 1e-4:
        raise RuntimeError(f"input_ids/inputs_embeds prefill mismatch: {result}")
    if result["cache_next_max_abs_logit_difference"] > 1e-4 or not result["next_token_equal"]:
        raise RuntimeError(f"input_ids/inputs_embeds cache mismatch: {result}")
    return result


def make_decoder(args, model, cache, device):
    if args.init_checkpoint:
        decoder, checkpoint = load_decoder(args.init_checkpoint, device)
        return decoder, {"initialized_from": args.init_checkpoint, "source_stage": checkpoint["stage"]}
    target_rms = evidence_embedding_rms(model, cache, min(args.max_samples, len(cache)), device)
    decoder = SoftPrefixDecoder(
        receiver_dim=model.config.hidden_size,
        max_tokens=args.max_evidence_tokens,
        target_rms=target_rms,
    ).to(device)
    return decoder, {"initialized_from": "random", "target_rms": target_rms}


def train_stage_a(model, decoder, cache, indices, optimizer, device, epoch, args):
    order = indices.copy()
    random.Random(args.seed + epoch).shuffle(order)
    losses, cosines, mses, gradients = [], [], [], []
    embedding = model.get_input_embeddings()
    for index in tqdm(order, desc=f"p3e_j_{args.mode}_stage_a_epoch{epoch}"):
        payload = cache.load(index)
        keys = payload["keys"].to(device)
        values = payload["values"].to(device)
        mask = payload["mask"].to(device)
        token_ids = payload["token_ids"].to(device)
        target = embedding(token_ids).detach().float()
        optimizer.zero_grad(set_to_none=True)
        predicted = decoder(keys, values, mask)
        loss, components = embedding_reconstruction_loss(predicted, target, mask)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite Stage A loss")
        loss.backward()
        assert_frozen_gradients(model)
        gradient = torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.grad_clip)
        if not torch.isfinite(gradient):
            raise RuntimeError("Non-finite Stage A gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
        cosines.append(float(1.0 - components["cosine_loss"].detach()))
        mses.append(float(components["mse"].detach()))
        gradients.append(float(gradient))
    return {
        "epoch": epoch,
        "loss": sum(losses) / len(losses),
        "embedding_cosine": sum(cosines) / len(cosines),
        "embedding_mse": sum(mses) / len(mses),
        "gradient_norm": sum(gradients) / len(gradients),
    }


def train_stage_b(model, decoder, cache, indices, optimizer, tokenizer, device, epoch, args):
    order = indices.copy()
    random.Random(args.seed + 1000 + epoch).shuffle(order)
    totals = {
        "loss": [], "answer_nll": [], "embedding_loss": [],
        "hidden_loss": [], "kl_loss": [], "gradient_norm": [],
    }
    embedding = model.get_input_embeddings()
    for index in tqdm(order, desc=f"p3e_j_{args.mode}_stage_b_epoch{epoch}"):
        payload = cache.load(index)
        keys = payload["keys"].to(device)
        values = payload["values"].to(device)
        mask = payload["mask"].to(device)
        token_ids = payload["token_ids"].to(device)
        target = embedding(token_ids).detach().float()
        optimizer.zero_grad(set_to_none=True)
        soft = decoder(keys, values, mask)
        embed_loss, _ = embedding_reconstruction_loss(soft, target, mask)
        batch = build_teacher_forcing_batch(
            model, tokenizer, payload["row"], token_ids, soft, device
        )
        if batch["ids"].shape[1] > args.max_sequence_length:
            raise RuntimeError(
                f"Sequence length {batch['ids'].shape[1]} exceeds {args.max_sequence_length}"
            )
        teacher = model(
            input_ids=batch["ids"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        student = model(
            inputs_embeds=batch["student_embeddings"],
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        answer_loss = answer_mean_nll(student.logits, batch["labels"])
        hidden_loss = hidden_alignment_loss(
            student.hidden_states, teacher.hidden_states,
            batch["question_mask"], args.hidden_layers,
        )
        kl_loss = answer_kl(
            student.logits, teacher.logits, batch["labels"], args.temperature
        )
        loss = (
            answer_loss
            + args.embed_weight * embed_loss
            + args.hidden_weight * hidden_loss
            + args.kl_weight * kl_loss
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite Stage B loss")
        loss.backward()
        assert_frozen_gradients(model)
        gradient = torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.grad_clip)
        if not torch.isfinite(gradient):
            raise RuntimeError("Non-finite Stage B gradient")
        optimizer.step()
        totals["loss"].append(float(loss.detach()))
        totals["answer_nll"].append(float(answer_loss.detach()))
        totals["embedding_loss"].append(float(embed_loss.detach()))
        totals["hidden_loss"].append(float(hidden_loss.detach()))
        totals["kl_loss"].append(float(kl_loss.detach()))
        totals["gradient_norm"].append(float(gradient))
        del teacher, student
    return {"epoch": epoch, **{
        name: sum(values) / len(values) for name, values in totals.items()
    }}


@torch.inference_mode()
def one_sample_reconstruction(model, decoder, payload, device):
    keys = payload["keys"].to(device)
    values = payload["values"].to(device)
    mask = payload["mask"].to(device)
    token_ids = payload["token_ids"].to(device)
    target = model.get_input_embeddings()(token_ids).float()
    predicted = decoder(keys, values, mask)
    return reconstruction_metrics(predicted, target, token_ids, mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["a", "b"], required=True)
    parser.add_argument("--mode", choices=["smoke16", "formal512"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--canonical-index", required=True)
    parser.add_argument("--native-index", required=True)
    parser.add_argument("--writer-checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--embed-weight", type=float, default=0.2)
    parser.add_argument("--hidden-weight", type=float, default=0.5)
    parser.add_argument("--kl-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[8, 16, 24, 35])
    parser.add_argument("--max-evidence-tokens", type=int, default=1024)
    parser.add_argument("--max-sequence-length", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cache = PairedCanonicalTokenCache(
        args.canonical_index, args.native_index, args.data, capacity=2
    )
    count = min(args.max_samples, len(cache))
    if args.mode == "formal512" and count != 512:
        raise RuntimeError("Formal training requires exactly 512 samples")
    model, tokenizer = load_receiver(args.model, device)
    for index in range(count):
        verify_tokenizer_alignment(tokenizer, cache.load(index))
    interface = exact_embedding_interface_check(
        model, tokenizer, cache.load(0), device
    )
    decoder, initialization = make_decoder(args, model, cache, device)
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    assert_optimizer_only_decoder(model, decoder, optimizer)
    initial_reconstruction = one_sample_reconstruction(
        model, decoder, cache.load(0), device
    )
    indices = list(range(count))
    history, best_loss, best_epoch = [], float("inf"), -1
    for epoch in range(1, args.epochs + 1):
        if args.stage == "a":
            summary = train_stage_a(
                model, decoder, cache, indices, optimizer, device, epoch, args
            )
        else:
            summary = train_stage_b(
                model, decoder, cache, indices, optimizer,
                tokenizer, device, epoch, args,
            )
        history.append(summary)
        torch.save(
            decoder_checkpoint(decoder, args.stage, epoch, history, args),
            output / f"checkpoint_epoch_{epoch}.pt",
        )
        if summary["loss"] < best_loss:
            best_loss, best_epoch = summary["loss"], epoch
            torch.save(
                decoder_checkpoint(decoder, args.stage, epoch, history, args),
                output / "checkpoint_best.pt",
            )
        write_jsonl(output / "training_history.jsonl", history)
    torch.save(
        decoder_checkpoint(decoder, args.stage, args.epochs, history, args),
        output / "checkpoint_last.pt",
    )
    final_reconstruction = one_sample_reconstruction(
        model, decoder, cache.load(0), device
    )
    result = {
        "status": "complete",
        "experiment": "P3-E-J Canonical Soft-Prefix Executability Diagnosis",
        "stage": args.stage,
        "mode": args.mode,
        "samples": count,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_training_loss": best_loss,
        "initialization": initialization,
        "interface_equivalence": interface,
        "initial_sample_reconstruction": initial_reconstruction,
        "final_sample_reconstruction": final_reconstruction,
        "decoder_metadata": decoder.metadata(),
        "trainable_parameters": sum(parameter.numel() for parameter in decoder.parameters()),
        "receiver_trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "receiver_frozen_gradient_audit": "passed_each_step",
        "canonical_index_sha256": sha256_file(args.canonical_index),
        "native_index_sha256": sha256_file(args.native_index),
        "fixed_c2_writer_checkpoint": args.writer_checkpoint,
        "fixed_c2_writer_sha256": sha256_file(args.writer_checkpoint),
        "data_sha256": sha256_file(args.data),
        "history": history,
    }
    write_json(output / "TRAIN_SUCCESS.json", result)


if __name__ == "__main__":
    main()
