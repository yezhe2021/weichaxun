# Gemma3-4B -> Qwen3-4B Full34 HeadMix256 Results

Protocol: `openbookqa_official_gemma3_4b_onboarding_full34_headmix256_options32_v1`; split: `4096/128/128`; seed `1234`.

## Native baselines and controls

| Condition | Correct | Accuracy |
|---|---:|---:|
| gemma3_full_native | 97 | 75.78% |
| gemma3_native_selected32 | 90 | 70.31% |
| qwen_full_native | 96 | 75.00% |
| qwen_native_oracle32 | 95 | 74.22% |
| qwen_self_selected32 | 94 | 73.44% |
| qwen_zero32 | 31 | 24.22% |
| qwen_no_memory | 33 | 25.78% |
| qwen_shuffled_oracle32 | 34 | 26.56% |

## Translator results

| Method | Accuracy | Oracle agreement | Choice KL | K cosine | V cosine | Oracle gap |
|---|---:|---:|---:|---:|---:|---:|
| stage_a | 40.62% | 42.19% | 1.036874 | 0.950766 | 0.778630 | 33.59 pp |
| residual | 74.22% | 70.31% | 0.549547 | 0.949200 | 0.777011 | 0.00 pp |
| shared | 73.44% | 65.62% | 0.629780 | 0.753621 | 0.691912 | 0.78 pp |

## Paired correctness

```json
{
  "paired_stage_a_vs_residual": {
    "both_correct": 46,
    "only_first_correct": 6,
    "only_second_correct": 49,
    "both_wrong": 27,
    "second_minus_first": 0.3359375,
    "mcnemar_p": 1.822834494458192e-09
  },
  "paired_stage_a_vs_shared": {
    "both_correct": 43,
    "only_first_correct": 9,
    "only_second_correct": 51,
    "both_wrong": 25,
    "second_minus_first": 0.328125,
    "mcnemar_p": 3.085035609438902e-08
  },
  "paired_residual_vs_shared": {
    "both_correct": 83,
    "only_first_correct": 12,
    "only_second_correct": 11,
    "both_wrong": 22,
    "second_minus_first": -0.0078125,
    "mcnemar_p": 1.0
  }
}
```

The machine-readable per-sample predictions, training traces, native controls, and learned head/layer correspondence matrices are stored under `runs/study/`.

## Data caveat

The predefined official subsets contain one semantic train/test duplicate under option permutation (`test 9-317`, `train 13-308`). Stage A, Residual64, Shared Stage B, and all native baselines answer this test item correctly. Excluding it gives Stage A `51/127 = 40.16%`, Residual64 `94/127 = 74.02%`, Shared Stage B `93/127 = 73.23%`, and Qwen Native Oracle32 `94/127 = 74.02%`. The main table keeps the predefined 128-item protocol for direct comparability with earlier onboarding experiments.
