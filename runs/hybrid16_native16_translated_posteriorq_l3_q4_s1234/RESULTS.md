# Hybrid 16 Native + 16 Translated KV Results

The evaluation completed on 128 MMLU-Pro test samples. It reused the posterior-
Question Sender experiment's selected anchors, Receiver protocol, and trained
Full28-Diagonal Translator. No retraining was performed.

## Existing controls

| Condition | Accuracy | Oracle agreement | Choice KL |
|---|---:|---:|---:|
| Full translated, best-accuracy checkpoint | 24.22% | 46.88% | 0.790903 |
| Full native selected-32 oracle | 33.59% | 100.00% | 0.000000 |
| Zero | 13.28% | 20.31% | 2.405776 |
| No memory | 13.28% | 25.00% | 2.777405 |

## Hybrid results

| Checkpoint | Native positions | Accuracy | Oracle agreement | Choice KL |
|---|---|---:|---:|---:|
| best accuracy, step 128 | even 0,2,...,30 | 18.75% | 36.72% | 0.917368 |
| best accuracy, step 128 | odd 1,3,...,31 | 19.53% | 37.50% | 0.932208 |
| best choice-KL, step 384 | even 0,2,...,30 | 14.84% | 32.81% | 1.045992 |
| best choice-KL, step 384 | odd 1,3,...,31 | 19.53% | 25.00% | 1.026180 |

Under both complementary parity patterns, mixing 16 position-matched native KV
states with 16 translated KV states did not interpolate toward the native oracle.
For the primary checkpoint, it reduced accuracy by 4.69--5.47 percentage points
relative to the 32-position fully translated cache. This is evidence of a joint
cache compatibility issue: native and translated per-position states are not
freely interchangeable even when their positions, mask, selected anchors, and
K/V pairing are held fixed.
