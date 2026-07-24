import argparse
from pathlib import Path

from p3e_f_common import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    evaluation = read_json(args.evaluation)
    metrics = evaluation["conditions"]
    reconstruction = evaluation["embedding_reconstruction"]
    cosine = reconstruction["embedding_cosine"]
    soft_f1 = metrics["canonical_soft_prefix"]["f1"]
    full_f1 = metrics["full_evidence_text"]["f1"]
    c1_f1 = metrics["current_c1_headwise_reader"]["f1"]
    reconstruction_good = cosine >= 0.80
    soft_near_text = soft_f1 >= full_f1 - 0.08
    soft_near_c1 = abs(soft_f1 - c1_f1) <= 0.05
    if reconstruction_good and soft_near_text:
        provisional = "canonical_content_sufficient_external_residual_interface_is_primary_gap"
    elif reconstruction_good and soft_near_c1:
        provisional = "embedding_similarity_does_not_restore_executable_reasoning_trajectory"
    elif not reconstruction_good:
        provisional = "canonical_memory_does_not_preserve_full_evidence_token_content"
    else:
        provisional = "mixed_result_requires_per_sample_and_manual_semantic_review"
    write_json(Path(args.out), {
        "status": "complete",
        "provisional_automatic_diagnosis": provisional,
        "manual_CPW_review_required_for_final_interpretation": True,
        "thresholds": {
            "embedding_cosine_good": 0.80,
            "soft_prefix_within_full_text_f1": 0.08,
            "soft_prefix_within_c1_f1": 0.05,
        },
        "observed": {
            "embedding_cosine": cosine,
            "canonical_soft_prefix_f1": soft_f1,
            "full_evidence_text_f1": full_f1,
            "current_c1_f1": c1_f1,
            "correct_shuffled_f1_gap": evaluation["correct_shuffled_f1_gap"],
        },
        "scope": (
            "Diagnostic evidence only. KV-to-soft-token conversion is not proposed "
            "as the final heterogeneous latent communication interface."
        ),
    })


if __name__ == "__main__":
    main()
