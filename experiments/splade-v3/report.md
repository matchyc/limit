# SPLADE-v3 on official LIMIT queries

Requested `naver/splade-v3`. Used `naver/splade-v3-distilbert` on cuda. BERT v3 gated=True. 66.6s.

Recall is macro over the two golds, in percent. Original `Who likes X?` queries.

## limit-small (46 documents)

| score | R@2 | R@10 | R@20 | both@2 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 99.15 | 100.0 | 100.0 | 98.4 |
| SPLADE-v3 whole | 92.45 | 99.6 | 99.85 | 86.4 |
| SPLADE-v3 chunk-max | 96.65 | 99.25 | 99.65 | 94.5 |
| SPLADE++ EnsembleDistil whole | 92.05 | 99.35 | 99.7 | 86.8 |

## limit (50k documents)

| score | R@2 | R@10 | R@20 | R@100 | both@2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 92.65 | 94.6 | 95.15 | 96.3 | 91.0 |
| SPLADE-v3 whole | 38.9 | 62.55 | 69.8 | 82.35 | 23.9 |
| SPLADE++ EnsembleDistil whole | 33.8 | 58.65 | 68.95 | 83.55 | 23.3 |
| paper GTE-ModernColBERT | 23.1 | 34.6 | — | 54.8 | — |

v3-DistilBERT MSMARCO MRR@10 is 38.7 vs BERT v3 40.2 vs ++ EnsembleDistil 38.0. The LIMIT comparison is against the same ++ checkpoint used earlier.

