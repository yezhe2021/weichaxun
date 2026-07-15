from sender_answerability_demo import compatible_negative, load_complete_pairs, negative_mapping


DATA = "/home/yezhe/伪查询/runs/p2a2_query_output_native_kv_qwen3_8b_seed1234/data/test.jsonl"

pairs = load_complete_pairs(DATA, max_pairs=16, seed=1234)
negatives = negative_mapping(pairs)
assert len(pairs) == len(negatives) == 16
assert all(compatible_negative(pair, negative) for pair, negative in zip(pairs, negatives))
assert all(pair["base"]["answer"] != pair["counterfactual"]["answer"] for pair in pairs)
print("p2d_data_smoke=passed")
