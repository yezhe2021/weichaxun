import torch

from choice_mass_audit import functional_distances, margin_buckets


def run():
    torch.manual_seed(1234)
    native, writer = torch.randn(97), torch.randn(97)
    ids = torch.tensor([2, 5, 7, 11, 19, 23, 29, 31, 37, 41])
    metrics = functional_distances(native, writer, ids)
    total = (metrics["choice_vs_nonchoice_mass_kl"] + metrics["weighted_choice_kl_contribution"]
             + metrics["nonchoice_conditional_kl_contribution"])
    assert abs(total - metrics["full_vocab_kl"]) < 1e-6
    shifted = functional_distances(native + 17.0, writer - 9.0, ids)
    assert abs(shifted["choice_only_kl"] - metrics["choice_only_kl"]) < 1e-5
    assert abs(shifted["centered_choice_logit_mse"] - metrics["centered_choice_logit_mse"]) < 1e-5
    rows = []
    for index in range(17):
        rows.append({"native": {"top1_top2_logit_margin": float(index)}, "checkpoints": {"x": {
            "native_top1_agreement": 1.0, "native_top_pair_flipped": 0.0,
            "choice_only_kl": 0.1, "centered_choice_logit_mse": 0.2,
        }}})
    assert sum(x["count"] for x in margin_buckets(rows, "x")) == len(rows)


if __name__ == "__main__":
    run()
