# Formal translated-like KV experiment

Receiver: Qwen3-0.6B. Data: HotpotQA, 512 train / 64 dev. Context limit: 256 tokens. 
All reported checkpoints use seed 1234 and one training epoch. Evaluation includes 16-token greedy generation.

## Equivalence guard

| Experiment | Within atol | Strict top1 | N | Max abs logit |
|---|---:|---:|---:|---:|
| autoencoder | 64 | 63 | 64 | 0.191406 |
| mse_only | 64 | 63 | 64 | 0.191406 |
| ce_only | 64 | 63 | 64 | 0.191406 |
| mse_ce | 64 | 63 | 64 | 0.191406 |
| rope_mse_ce | 64 | 63 | 64 | 0.191406 |
| deterministic_controls | 64 | 63 | 64 | 0.191406 |

## Training validation

| Experiment | Val relative MSE | Val CE | MSE weight |
|---|---:|---:|---:|
| autoencoder | 0.3172 | 11.3032 |  |
| mse_only | 0.4069 | 12.7412 |  |
| ce_only | 134.2637 | 7.4807 |  |
| mse_ce | 1.2215 | 3.6873 | 1.0000 |
| rope_mse_ce | 0.9041 | 8.4339 | 1.3953 |

## Direct replacement

| Experiment | Rel MSE | K cos | V cos | Logit KL | CE delta | Top1 | Route | Attn out cos | KV joint | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| autoencoder | 0.3172 | 0.9047 | 0.6853 | 6.9463 | 6.6172 | 0.1225 | 0.2950 | 0.1244 | 0.6106 | 0.0105 |
| ce_only | 134.2648 | 0.0037 | 0.0155 | 6.2891 | 2.7943 | 0.0520 | 0.0868 | 0.0152 | -0.0002 | 0.0000 |
| mse_ce | 1.2215 | 0.5126 | 0.1201 | 4.2952 | -0.9967 | 0.3444 | 0.0961 | 0.0130 | 0.0423 | 0.0985 |
| mse_only | 0.4069 | 0.8775 | 0.5955 | 8.7602 | 8.0574 | 0.0659 | 0.2677 | 0.1244 | 0.5463 | 0.0000 |
| rope_mse_ce | 0.9041 | 0.7323 | 0.2704 | 6.4021 | 3.7469 | 0.0134 | 0.0894 | -0.0196 | 0.1532 | 0.0078 |

## Residual fusion

| Experiment | Alpha | CE delta | Logit KL | Top1 | Attn out cos | KV joint | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| autoencoder | 0.25 | -0.3604 | 1.3916 | 0.4880 | 0.6999 | 0.9857 | 0.0771 |
| autoencoder | 0.50 | 1.5672 | 2.7598 | 0.2803 | 0.4315 | 0.9318 | 0.0377 |
| autoencoder | 0.75 | 4.0177 | 4.5359 | 0.1596 | 0.2421 | 0.8179 | 0.0132 |
| autoencoder | 1.00 | 6.6172 | 6.9463 | 0.1225 | 0.1244 | 0.6106 | 0.0105 |
| ce_only | 0.25 | 8.1029 | 10.3122 | 0.0234 | 0.3754 | 0.8763 | 0.0126 |
| ce_only | 0.50 | 3.4651 | 7.2359 | 0.0276 | 0.1357 | 0.6871 | 0.0000 |
| ce_only | 0.75 | 2.8148 | 6.4431 | 0.0590 | 0.0551 | 0.3240 | 0.0000 |
| ce_only | 1.00 | 2.7943 | 6.2891 | 0.0520 | 0.0152 | -0.0002 | 0.0000 |
| mse_ce | 0.25 | 1.6999 | 3.2805 | 0.2991 | 0.6669 | 0.9732 | 0.0644 |
| mse_ce | 0.50 | 0.1936 | 3.1168 | 0.3523 | 0.2403 | 0.8499 | 0.0638 |
| mse_ce | 0.75 | -1.2529 | 2.9856 | 0.4336 | 0.0657 | 0.5206 | 0.0736 |
| mse_ce | 1.00 | -0.9967 | 4.2952 | 0.3444 | 0.0130 | 0.0423 | 0.0985 |
| mse_only | 0.25 | 0.7670 | 1.9080 | 0.4342 | 0.7286 | 0.9838 | 0.0780 |
| mse_only | 0.50 | 4.6845 | 5.4359 | 0.1786 | 0.4351 | 0.9190 | 0.0482 |
| mse_only | 0.75 | 6.1827 | 6.4831 | 0.1021 | 0.2365 | 0.7794 | 0.0273 |
| mse_only | 1.00 | 8.0574 | 8.7602 | 0.0659 | 0.1244 | 0.5463 | 0.0000 |
| rope_mse_ce | 0.25 | 3.3909 | 3.6946 | 0.2239 | 0.4369 | 0.9684 | 0.0437 |
| rope_mse_ce | 0.50 | 4.3230 | 6.3665 | 0.0152 | 0.0993 | 0.8541 | 0.0104 |
| rope_mse_ce | 0.75 | 3.7305 | 6.3088 | 0.0134 | 0.0184 | 0.5871 | 0.0000 |
| rope_mse_ce | 1.00 | 3.7469 | 6.4021 | 0.0134 | -0.0196 | 0.1532 | 0.0078 |

## Findings

- All 64 samples pass the FP16 absolute-logit tolerance. One near-tied sample changes one argmax, so strict full/split top-1 equivalence is 63/64.
- CE-only moves toward a functional shortcut during training: validation CE is 7.4807, while validation relative MSE explodes to 134.2637. Its direct generated F1 is nevertheless 0.0000, so CE-only has not learned a robust readable cache.
- MSE-only achieves direct relative MSE 0.4069, but direct CE delta remains 8.0574; KV fitting alone does not ensure receiver readability.
- MSE+CE is the strongest functional-shortcut result: direct CE delta -0.9967, F1 0.0985 versus native 0.1068, despite attention-output cosine 0.0130 and KV-joint consistency 0.0423.
- RoPE disentangling does not help in this setup: direct CE delta is 3.7469 versus -0.9967 for ordinary MSE+CE.
- Increasing the native-cache share consistently restores KV geometry and attention readout, but CE/F1 can be non-monotonic for CE-trained translators. This decoupling is itself evidence that task loss can exploit cache shortcuts rather than reconstruct native memory.
- Joint token permutation is a calibration control: route overlap changes sharply while attention output can remain invariant, so routing metrics must be interpreted jointly with readout and logits.

These results support a receiver-specific structured-memory interpretation for native prefill KV within this model and pseudo-translation setup. They do not by themselves establish the behavior of every real heterogeneous translator.
