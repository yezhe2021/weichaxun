import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from p3d3_common import file_sha256, load_receiver, write_json
from p3e_n_common import ReceiverNativeConditionedCache
from p3e_p_common import (
    SELECTED_LAYERS,
    answer_suffix,
    full_text_prompt,
    prediction_positions,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--memory", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    cache = ReceiverNativeConditionedCache(args.memory)
    count = min(args.max_samples, len(cache))
    model, tokenizer = load_receiver(args.model, torch.device(args.device))
    entries = []

    for index in tqdm(range(count), desc="p3e_p_cache_teacher"):
        row = cache.correct(index)["row"]
        prompt_ids = tokenizer(
            full_text_prompt(tokenizer, row), add_special_tokens=False
        ).input_ids
        suffix = answer_suffix(tokenizer, row["answer"])
        ids = torch.tensor(
            [prompt_ids + suffix], dtype=torch.long, device=model.device
        )
        positions = prediction_positions(
            len(prompt_ids), len(suffix), model.device
        )
        captured = {}
        handles = []
        for layer_index in SELECTED_LAYERS:
            layer = model.model.layers[layer_index]

            def hook(module, args, kwargs, layer_index=layer_index):
                captured[layer_index] = (
                    args[0][:, positions].detach().to(torch.float16).cpu()
                )

            handles.append(
                layer.post_attention_layernorm.register_forward_pre_hook(
                    hook, with_kwargs=True
                )
            )
        try:
            with torch.inference_mode():
                model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    position_ids=torch.arange(
                        ids.shape[1], device=model.device
                    ).unsqueeze(0),
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()
        states = torch.stack(
            [captured[layer][0] for layer in SELECTED_LAYERS]
        )
        payload = {
            "id": row["id"],
            "answer": row["answer"],
            "answer_token_ids": suffix,
            "teacher_states": states,
            "selected_layers": SELECTED_LAYERS,
            "teacher_prompt_length": len(prompt_ids),
            "prediction_indices": positions.cpu().tolist(),
            "teacher_visible_answer_prefix_lengths": list(range(len(suffix))),
            "future_answer_visible": False,
        }
        filename = f"sample_{index:05d}.pt"
        torch.save(payload, output / filename)
        entries.append(
            {
                "id": row["id"],
                "file": filename,
                "answer_tokens": len(suffix),
            }
        )

    write_json(
        output / "index.json",
        {
            "status": "complete",
            "experiment": "P3-E-P frozen full-text teacher trajectory cache",
            "samples": count,
            "selected_layers": SELECTED_LAYERS,
            "state": "post_attention current prediction token only",
            "gold_current_or_future_answer_visible": False,
            "entries": entries,
        },
    )
    write_json(
        output / "SUCCESS.json",
        {
            "status": "complete",
            "samples": count,
            "index_sha256": file_sha256(output / "index.json"),
        },
    )


if __name__ == "__main__":
    main()

