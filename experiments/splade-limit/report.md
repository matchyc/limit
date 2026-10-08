# SPLADE++ on official LIMIT queries

Model `naver/splade-cocondenser-ensembledistil` on cuda. Original `Who likes X?` queries. limit and limit-small are different draws. 84.7s.

Recall is macro over the two golds, in percent.

## limit-small (46 documents)

| score | R@2 | R@10 | R@20 | both@2 |
| --- | ---: | ---: | ---: | ---: |
| BM25 (ours) | 99.15 | 100.0 | 100.0 | 98.4 |
| SPLADE whole | 92.05 | 99.35 | 99.7 | 86.8 |
| SPLADE chunk-max | 99.75 | 100.0 | 100.0 | 99.5 |
| paper BM25 | 97.8 | 100.0 | — | — |
| paper GTE-ModernColBERT | 83.5 | 97.6 | — | — |

## limit (50k documents)

| score | R@2 | R@10 | R@20 | R@100 |
| --- | ---: | ---: | ---: | ---: |
| BM25 (ours) | 92.65 | 94.6 | 95.15 | 96.3 |
| SPLADE whole | 33.8 | 58.65 | 68.95 | 83.55 |
| paper BM25 | 85.7 | 90.4 | — | 93.6 |
| paper GTE-ModernColBERT | 23.1 | 34.6 | — | 54.8 |
| paper Promptriever 4096 | 3.0 | 6.8 | — | 18.9 |

