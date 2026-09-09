# P3-E-L Question-Conditioned Sender Diagnosis

This diagnostic keeps the Qwen3-4B Receiver, C1 Headwise Reader, C2 canonical
format, data split, prompt, and decoding fixed. Only the Qwen3-8B Sender prefill
context changes.

The transmitted memory always contains Evidence-token K/V only:

- `evidence_only`: existing `M(E)` baseline.
- `neutral_prefix`: length-matched nonsemantic prefix before Evidence.
- `wrong_question`: type- and length-matched wrong Question before Evidence.
- `correct_question`: current Question before Evidence, producing `M(E|Q)`.
- `correct_question_hard_shuffled_evidence`: current Question with hard-negative
  Evidence.

The workflow first performs zero-shot evaluation with the frozen C2 Writer. An
automatic predeclared F1 rule decides whether to initialize a new Writer from C2
and train only that Writer for 8 epochs. The completed run used six initial
epochs plus a two-epoch continuation from the best checkpoint. The C1 Reader and Receiver remain
frozen. Final evaluation writes all generations and a manual C/P/W worksheet.
