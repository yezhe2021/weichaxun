# P3-E-N2 Reader Readout Bottleneck Diagnosis

This experiment freezes the Qwen3-4B backbone and the trained P3-E-N Reader C.
For correct and hard-shuffled memory it caches:

- the 16 per-layer post-native-`o_proj`, pre-gate Reader readouts at the final
  Question prompt token: `[16,2560]`;
- the complete Reader token attention without layer/head averaging:
  `[16,32,T]`.

A standalone probe receives only these Reader outputs, never Native K/V. It
predicts supporting tokens, extractive answer spans, and yes/no answers.
Correct, shuffled-source, and zero controls determine whether Reader C
retrieved evidence content before residual injection into the frozen Receiver.
Shuffled predictions are scored against the source-memory labels.
