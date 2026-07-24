import argparse

from p3e_k_common import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    evaluation = read_json(args.evaluation)
    conditions = evaluation["conditions"]
    text = conditions["full_text_representation"]
    native = conditions["sender_native_kv"]
    canonical = conditions["learned_canonical_kv"]
    support = canonical["support_sentence_recall_at_2"] or 0.0
    span = canonical["span_token_f1"] or 0.0
    native_support = native["support_sentence_recall_at_2"] or 0.0
    native_span = native["span_token_f1"] or 0.0
    text_support = text["support_sentence_recall_at_2"] or 0.0
    text_span = text["span_token_f1"] or 0.0
    if support >= 0.70 and span < 0.40:
        diagnosis = "D_supporting_facts_present_but_answer_relation_not_decoded"
    elif (
        support >= 0.8 * min(native_support, text_support)
        and span >= 0.8 * min(native_span, text_span)
    ):
        diagnosis = "A_canonical_information_sufficient_receiver_execution_is_primary_gap"
    elif native_support - support >= 0.15 or native_span - span >= 0.15:
        diagnosis = "B_native_contains_information_but_canonical_writer_loses_it"
    elif (
        text_support - max(native_support, support) >= 0.15
        or text_span - max(native_span, span) >= 0.15
    ):
        diagnosis = "C_selected_native_and_canonical_KV_are_not_sufficient_for_probe"
    else:
        diagnosis = "mixed_result_requires_per_sample_review"
    write_json(args.out, {
        "status": "complete",
        "provisional_diagnosis": diagnosis,
        "diagnostic_only": True,
        "observed": {
            "full_text": {
                "support_sentence_recall_at_2": text_support,
                "span_token_f1": text_span,
            },
            "native": {
                "support_sentence_recall_at_2": native_support,
                "span_token_f1": native_span,
            },
            "canonical": {
                "support_sentence_recall_at_2": support,
                "span_token_f1": span,
            },
            "canonical_correct_minus_sample_shuffled_answer_f1": (
                evaluation["canonical_correct_minus_sample_shuffled_current_answer_f1"]
            ),
            "canonical_correct_minus_hard_shuffled_answer_f1": (
                evaluation["canonical_correct_minus_hard_shuffled_current_answer_f1"]
            ),
        },
    })


if __name__ == "__main__":
    main()
