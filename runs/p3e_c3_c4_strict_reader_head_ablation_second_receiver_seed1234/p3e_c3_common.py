import torch


def initialize_fresh_reader(reader, mode, seed, gate_init=0.01):
    if mode not in ("fully_random", "weak_pair"):
        raise ValueError(f"Unknown initialization mode: {mode}")
    with torch.no_grad():
        for layer_index, branch in enumerate(reader.branches):
            generator = torch.Generator(device=branch.query_adapter.up.device).manual_seed(int(seed) + layer_index * 1009)
            for head in range(branch.query_adapter.heads):
                torch.nn.init.orthogonal_(branch.query_adapter.down[head])
            branch.query_adapter.up.normal_(mean=0.0, std=1e-3, generator=generator)
            branch.gate.fill_(float(gate_init))
            logits = torch.randn(branch.router_logits.shape, generator=generator, device=branch.router_logits.device) * (0.02 if mode == "fully_random" else 0.05)
            if mode == "weak_pair":
                for query_head in range(branch.query_heads):
                    pair = 2 * (query_head // 4)
                    logits[query_head, pair:pair + 2] += 0.5
            branch.router_logits.copy_(logits)
    return {
        "mode": mode,
        "seed": int(seed),
        "query_adapter": "fresh orthogonal down plus Normal(0,1e-3) up",
        "route": "Normal(0,0.02)" if mode == "fully_random" else "Normal(0,0.05) plus 0.5 preferred-pair bias",
        "gate": float(gate_init),
        "native_reader_loaded": False,
        "c1_reader_loaded": False,
    }


def intervene_pair(memory, pair_index, mode):
    if not 0 <= pair_index < 8: raise ValueError("pair_index must be in [0,7]")
    if mode not in ("both", "first_only", "second_only", "swap", "copy_first_to_second", "copy_second_to_first"):
        raise ValueError(f"Unknown intervention: {mode}")
    first, second = 2 * pair_index, 2 * pair_index + 1
    result = dict(memory); result["keys"] = memory["keys"].clone(); result["values"] = memory["values"].clone()
    for name in ("keys", "values"):
        value = result[name]
        if mode == "first_only": value[:, :, second].zero_()
        elif mode == "second_only": value[:, :, first].zero_()
        elif mode == "swap":
            saved = value[:, :, first].clone(); value[:, :, first].copy_(value[:, :, second]); value[:, :, second].copy_(saved)
        elif mode == "copy_first_to_second": value[:, :, second].copy_(value[:, :, first])
        elif mode == "copy_second_to_first": value[:, :, first].copy_(value[:, :, second])
    result["intervention"] = {"pair": [first, second], "mode": mode}
    return result


__all__ = ["initialize_fresh_reader", "intervene_pair"]
