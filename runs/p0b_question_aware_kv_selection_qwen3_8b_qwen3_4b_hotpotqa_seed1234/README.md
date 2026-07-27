# P0-B: Question-Aware KV Selection

This standalone continuation removes the gold-evidence oracle from P0-A while
keeping the Shared Writer, Query Adapters, and P0-A2 content probes completely
frozen. There is no Receiver and no answer generation.

Input order remains `Context → Question`. Each sender independently ranks
context sentences using its true native post-RoPE attention:

`mean(layer, query-head, question-token) sum(sentence tokens) softmax(QK^T)`

The selector uses all 32 native query heads and the same fixed layers as P0-A:
`[0, 5, 10, 15, 20, 25, 30, 35]`. Selection is sentence-level with Top-2,
Top-4, and Top-8 budgets. Gold supporting facts never participate in automatic
selection; they are used only for evaluation and for the Gold control groups.
Writer inputs remain pre-RoPE K plus native V.

Controls:

- Gold only
- Gold plus random distractors to budget 4
- Auto Top-2 / Top-4 / Top-8
- Random-4

K is evaluated against the full-context gold supporting sentences for all
AA/AB/BA/BB combinations; unselected gold evidence remains a miss. V uses the
frozen P0-A2 probes and reports same-context supporting-sentence metrics for
self, cross, and wrong-sample shuffled Canonical memory. Primary metrics are
Recall@5 and AUPRC. The report also includes native selector recall and A/B
selection Jaccard overlap.

The pipeline first runs a four-example validation smoke and then the fixed
64-example validation evaluation. It does not retrain the Selector or Writer.
