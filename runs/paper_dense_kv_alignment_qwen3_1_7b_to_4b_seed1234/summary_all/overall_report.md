# Qwen3-1.7B -> Qwen3-4B Dense KV Alignment Summary

Generated from existing train/eval result files. No model inference is run by this packaging script.

## Train Summary

| method_group | method | stage_name | global_step | val_rec | val_aware_ce | val_unaware_ce | val_mixed_gen |
| --- | --- | --- | --- | --- | --- | --- | --- |
| paper | paper_rec_then_mixed_generation | phase2_mixed_generation | 1024.0000 | 2.1007 | 3.8016 | 4.7609 | 4.2812 |
| baseline | mse_only | mse | 1024.0000 | 1.7759 | 8.8055 | 13.6451 | 11.2253 |
| baseline | mse_then_ce | ce | 1024.0000 | 2.0817 | 4.2590 | 4.7712 | 4.5151 |
| ours | q_aware_functional | phase2_q_aware_functional | 1024.0000 | 2.0943 | 4.0365 | 4.5224 | 4.2794 |

## HOTPOTQA Evaluation

| method_group | method | receiver_prompt_mode | translated_ce | ce_delta | top1_match | answer_f1 | final_answer_exact_match | logit_kl | route_overlap | readout_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | mse_only | context_aware | 8.8055 | 2.5713 | 0.3681 | 0.1678 | 0.0156 | 2.6047 | 0.5265 | 0.7717 |
| baseline | mse_only | context_unaware | 13.6451 | 8.0025 | 0.0968 | 0.0314 | 0.0000 | 7.7150 | 0.5412 | 0.7388 |
| baseline | mse_then_ce | context_aware | 4.2590 | -1.9751 | 0.3787 | 0.2407 | 0.0625 | 2.7695 | 0.4441 | 1.0681 |
| baseline | mse_then_ce | context_unaware | 4.7711 | -0.8714 | 0.2397 | 0.1488 | 0.0781 | 3.7734 | 0.4559 | 1.0057 |
| paper | paper_rec_then_mixed_generation | context_aware | 3.8016 | -2.4326 | 0.4069 | 0.2527 | 0.0469 | 2.8486 | 0.4407 | 1.0647 |
| paper | paper_rec_then_mixed_generation | context_unaware | 4.7609 | -0.8817 | 0.2424 | 0.1228 | 0.0469 | 3.9496 | 0.4539 | 1.0051 |
| ours | q_aware_functional | context_aware | 4.0365 | -2.1977 | 0.5404 | 0.2213 | 0.0156 | 0.8316 | 0.4469 | 0.9005 |
| ours | q_aware_functional | context_unaware | 4.5224 | -1.1202 | 0.4288 | 0.1279 | 0.0156 | 1.3284 | 0.4626 | 0.8654 |

## GSM8K Evaluation

| method_group | method | receiver_prompt_mode | translated_ce | ce_delta | top1_match | answer_f1 | final_answer_exact_match | logit_kl | route_overlap | readout_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | mse_only | context_aware | 1.6260 | 0.4932 | 0.7432 | 0.6675 | 0.5156 | 0.8212 | 0.6371 | 0.8879 |
| baseline | mse_only | context_unaware | 2.8527 | 1.6562 | 0.5659 | 0.5740 | 0.3906 | 2.2250 | 0.6395 | 0.8900 |
| baseline | mse_then_ce | context_aware | 1.0226 | -0.1102 | 0.9390 | 0.7763 | 0.9531 | 0.1060 | 0.5835 | 1.1448 |
| baseline | mse_then_ce | context_unaware | 1.6571 | 0.4606 | 0.7294 | 0.6344 | 0.8750 | 0.9653 | 0.5847 | 1.1276 |
| paper | paper_rec_then_mixed_generation | context_aware | 1.4097 | 0.2769 | 0.7724 | 0.6711 | 0.1094 | 0.6610 | 0.5804 | 1.1473 |
| paper | paper_rec_then_mixed_generation | context_unaware | 2.1238 | 0.9273 | 0.6371 | 0.5784 | 0.2500 | 1.4630 | 0.5817 | 1.1305 |
| ours | q_aware_functional | context_aware | 1.3419 | 0.2090 | 0.7780 | 0.6777 | 0.4844 | 0.6149 | 0.5843 | 1.0139 |
| ours | q_aware_functional | context_unaware | 1.5059 | 0.3094 | 0.7660 | 0.6588 | 0.9375 | 0.8007 | 0.5862 | 1.0072 |

## Output Files

- `eval_all_datasets.csv`: long-form evaluation table across HotpotQA and GSM8K.
- `train_summary.csv`: training-stage validation summary.
- `overall_wide_summary.csv`: one row per method with train, HotpotQA, and GSM8K key metrics.
- `overall_report.md`: this readable summary.
