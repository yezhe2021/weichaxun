# P0-A2: V Content Sufficiency

This is the final functional V gate for P0-A. It reuses the frozen Raw,
Private, and Shared Canonical representations from the preceding P0-A
experiment. It does not retrain Canonical, introduce a Receiver, predict
HotpotQA answers, or change the architecture.

For sender A, attention is fixed as:

`A_A = softmax(Q_A K_A^T)`

The experiment compares `A_A V_A`, `A_A V_B`, and `A_A V_B_wrong`. Sender B is
evaluated symmetrically. Thus K and attention are held fixed and only V changes.
Wrong-sample V is selected deterministically from the closest-length different
sample, then truncated or zero-padded to the target token length.

Every context sentence is embedded offline by a completely frozen Qwen3-4B
text encoder using masked mean pooling of the final hidden state. A single
linear probe maps a 128D canonical readout into the text-embedding dimension.
Probe A is trained only on A self readouts and then frozen before A→B transfer;
Probe B is trained only on B self readouts and then frozen before B→A transfer.

Evaluation retrieves HotpotQA gold supporting sentences only among sentences
from the same context. It reports support Recall@1/5, Hit@1/5, MRR, and AUPRC
for self, cross-V, and shuffled-V conditions across Raw, Private, and Shared
representations.

The pipeline first executes an 8-train/4-validation smoke test and proceeds to
the fixed 256-train/64-validation formal measurement only if all code paths
complete with finite losses and metrics.
