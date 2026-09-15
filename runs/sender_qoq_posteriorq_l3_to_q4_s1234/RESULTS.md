<!-- BEGIN mmlupro_sender_qoq_posterior_question_rawanchor_full28_diagonal_llama3_2_3b_to_qwen3_4b_seed1234 -->
## mmlupro_sender_qoq_posterior_question_rawanchor_full28_diagonal_llama3_2_3b_to_qwen3_4b_seed1234

Machine-generated after successful completion. Sender routing uses Question -> Options -> repeated Question; Receiver remains native Question -> translated Options KV -> native Answer:.

| Protocol | Accuracy | Oracle agreement | Oracle choice-KL |
|---|---:|---:|---:|
| Qwen Full Native | 41.41% | 64.06% | 0.622540 |
| Llama Full Native | 32.03% | 34.38% | 0.988471 |
| Qwen Self-Selected32 Native | 35.16% | 77.34% | 0.522998 |
| Llama-selected -> Qwen Native Oracle | 33.59% | 100.00% | 0.000000 |
| Translated Full28-Diagonal | 24.22% | 46.88% | 0.790903 |
| Shuffle | 11.72% | 18.75% | 1.220713 |
| Zero | 13.28% | 20.31% | 2.405776 |
| No-memory | 13.28% | 25.00% | 2.777405 |

Selection gap=7.81 points; translation gap=9.38 points; information gain vs zero=10.94 points; information gain vs no-memory=10.94 points.
<!-- END mmlupro_sender_qoq_posterior_question_rawanchor_full28_diagonal_llama3_2_3b_to_qwen3_4b_seed1234 -->
