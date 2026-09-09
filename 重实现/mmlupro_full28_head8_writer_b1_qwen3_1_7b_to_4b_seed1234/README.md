# Full-28 + Full-Head Writer (1.7B→4B)

The frozen Full-28 depth projection is initialized from the prior Stage-A best checkpoint.
For every target layer and target KV head, an independent bias-free `Linear(8*128,128)` mixes all
eight depth-projected source heads. K/V are separate, tokens are never mixed, and only head projections train.

Identity head initialization exactly embeds the old Full-28 Writer. The pipeline hard-gates K/V and final-logit
equivalence before Stage A, then runs the unchanged KV reconstruction and final-position full-vocabulary KL stages.

```bash
bash launch_experiments.sh
tail -f logs/full_pipeline.log
```
