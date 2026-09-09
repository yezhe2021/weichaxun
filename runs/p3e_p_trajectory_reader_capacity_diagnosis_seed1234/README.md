# P3-E-P Trajectory-Supervised Memory Interface Diagnosis

This experiment tests whether a receiver-specific interface can reconstruct
the frozen Qwen3-4B trajectory produced by real full-text evidence.

The same-model setup uses Qwen3-4B question-conditioned Native KV, 512 training
samples, 64 validation samples, and Reader C's 16 layers. Three readers are
trained independently for eight epochs:

- A0: single native-GQA AV readout plus shared state corrector, answer loss.
- A1: the same interface with `answer + 0.5 * normalized state MSE`.
- C1: four latent queries, two full-KV reads, and the same trajectory loss.

Teacher states are frozen full-text post-attention states at the token that
predicts each gold answer token. Causal teacher forcing guarantees that state
only sees the preceding answer prefix. Student answer positions are shifted by
the evidence-length position gap and audited to equal Teacher positions.

Validation generation is fully free-running and never runs the Teacher branch.
Cached validation Teacher states are read only in a separate teacher-forced
state-diagnostic pass. The Receiver backbone, native projections, LM head, and
Native KV remain frozen; optimizers contain only reader parameters.
