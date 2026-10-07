# Scheme failure measurements (LIMIT-small, MiniLM-L6-v2)

Recall is macro-averaged `|top-k ∩ gold| / 2` on a 0–100 scale. Ties break toward the alphabetically earlier document id. `data/limit` distractors are a different draw (seed-0 permutation), not the official negatives.

## Reproduction

| scoring | R@2 | R@10 | R@20 | both-in-top2 |
| --- | ---: | ---: | ---: | ---: |
| whole-doc | 16.2 | 48.05 | 69.45 | 2.3 |
| chunk-max | 98.15 | 100.0 | 100.0 | 96.4 |
| chunk-avg | 5.9 | 28.75 | 52.95 | 0.5 |
| legacy-split-chunk-max | 98.1 | 100.0 | 100.0 | 96.3 |
| bm25 | 99.15 | 100.0 | 100.0 | 98.4 |

## Query wording, all 1000 queries, documents unchanged

| query | BM25 R@2 | whole-doc R@2 | chunk-MaxP R@2 | chunk-AvgP R@2 |
| --- | ---: | ---: | ---: | ---: |
| original | 99.15 | 16.2 | 98.15 | 5.9 |
| enjoys | 99.15 | 16.4 | 97.8 | 6.2 |
| person-enjoys | 98.25 | 16.9 | 96.85 | 6.55 |
| attr-only | 99.15 | 18.75 | 98.85 | 6.3 |
| drop-attribute | 4.5 | 4.5 | 4.5 | 4.5 |

## Glosses that share no content token with the attribute

Subset size 126. Strict lexical cutoff after doc rewrite: 78 queries whose original content tokens have document frequency 0.

| setting | n | BM25 R@2 | whole-doc R@2 | chunk-MaxP R@2 |
| --- | ---: | ---: | ---: | ---: |
| original-wording | 126 | 96.83 | 15.48 | 96.83 |
| query-gloss-docs-exact | 126 | 3.57 | 7.94 | 24.21 |
| query-gloss-only-docs-exact | 126 | 3.57 | 9.13 | 25.79 |
| query-exact-docs-gloss | 126 | 4.37 | 16.27 | 17.46 |
| both-gloss | 126 | 100.0 | 35.71 | 100.0 |
| strict-query-gloss-docs-exact | 78 | 3.85 | 7.69 | 23.08 |
| strict-query-exact-docs-gloss | 78 | 3.21 | 16.03 | 21.15 |
| strict-both-gloss | 78 | 100.0 | 33.97 | 100.0 |

## Chunk geometry

| chunks | intact gold spans | R@2 | R@10 | both-in-top2 |
| --- | ---: | ---: | ---: | ---: |
| k1 | 100.0 | 98.15 | 100.0 | 96.4 |
| k2 | 100.0 | 80.4 | 97.3 | 65.4 |
| k4 | 100.0 | 47.7 | 78.5 | 21.2 |
| k8 | 100.0 | 29.05 | 60.2 | 7.4 |
| k16 | 100.0 | 18.95 | 50.7 | 3.5 |
| all | 100.0 | 16.2 | 48.05 | 2.3 |
| win8-stride4 | 100.0 | 64.05 | 90.35 | 44.1 |
| win8-stride4-offset3 | 100.0 | 68.0 | 93.25 | 48.8 |
| win16-stride8 | 100.0 | 44.2 | 75.55 | 19.5 |
| win32-stride16 | 100.0 | 31.2 | 62.95 | 9.1 |
| win10-nongap | 94.3 | 47.5 | 78.4 | 23.7 |
| random-win10 | 93.2 | 46.6 | 80.6 | 23.7 |
| whole-doc | 100.0 | 16.2 | 48.05 | 2.3 |
| chunk-avg-k1 | 100.0 | 5.9 | 28.75 | 0.5 |

## Late interaction proxy

- pooling: Pooling
- pool-module-agreement: 1.0
- token-cosine-mean-random-pairs: 0.1093
- mean-span-tokens: 2.4
- maxsim-sum-all-tokens: R@2 97.9, R@10 100.0, both-in-top2 96.0
- maxsim-sum-content-tokens: R@2 98.65, R@10 100.0, both-in-top2 97.5
- maxsim-best-content-token: R@2 82.75, R@10 98.8, both-in-top2 72.8
- maxsim-content-on-gloss-queries: R@2 14.68, R@10 55.56, both-in-top2 3.17
- late-chunk-max: R@2 98.8, R@10 100.0, both-in-top2 97.8
- late-chunk-max-on-gloss-queries: R@2 21.43, R@10 62.7, both-in-top2 13.49

## Fusion ceiling (RRF k=60)

- exact-bm25-whole: bm25 R@2 99.15, whole-doc R@2 16.2, union R@2 99.3, both-miss slots 14, queries blind at 2: 0, full-list RRF R@2 47.45, RRF-on-union R@2 66.1
- exact-bm25-chunkmax: bm25 R@2 99.15, chunk-max R@2 98.15, union R@2 99.95, both-miss slots 1, queries blind at 2: 0, full-list RRF R@2 99.55, RRF-on-union R@2 99.5
- exact-bm25-maxsim: bm25 R@2 99.15, maxsim-content R@2 98.65, union R@2 99.65, both-miss slots 7, queries blind at 2: 0, full-list RRF R@2 99.05, RRF-on-union R@2 99.05
- exact-whole-chunkmax: whole-doc R@2 16.2, chunk-max R@2 98.15, union R@2 98.4, both-miss slots 32, queries blind at 2: 0, full-list RRF R@2 39.9, RRF-on-union R@2 49.4
- drop-bm25-whole: bm25 R@2 4.5, whole-doc R@2 4.5, union R@2 9.0, both-miss slots 1820, queries blind at 2: 826, full-list RRF R@2 4.5, RRF-on-union R@2 4.5
- drop-bm25-chunkmax: bm25 R@2 4.5, chunk-max R@2 4.5, union R@2 9.0, both-miss slots 1820, queries blind at 2: 826, full-list RRF R@2 4.5, RRF-on-union R@2 4.5
- gloss-query-bm25-whole: bm25 R@2 3.57, whole-doc R@2 7.94, union R@2 11.51, both-miss slots 223, queries blind at 2: 99, full-list RRF R@2 8.33, RRF-on-union R@2 6.75
- gloss-query-bm25-chunkmax: bm25 R@2 3.57, chunk-max R@2 24.21, union R@2 26.59, both-miss slots 185, queries blind at 2: 74, full-list RRF R@2 14.29, RRF-on-union R@2 13.49
- gloss-query-bm25-maxsim: bm25 R@2 3.57, maxsim-content R@2 14.68, union R@2 17.06, both-miss slots 209, queries blind at 2: 89, full-list RRF R@2 8.33, RRF-on-union R@2 8.73
- docs-gloss-bm25-chunkmax: bm25 R@2 4.37, chunk-max R@2 17.46, union R@2 20.63, both-miss slots 200, queries blind at 2: 85, full-list RRF R@2 9.13, RRF-on-union R@2 10.32
- both-gloss-bm25-chunkmax: bm25 R@2 100.0, chunk-max R@2 100.0, union R@2 100.0, both-miss slots 0, queries blind at 2: 0, full-list RRF R@2 100.0, RRF-on-union R@2 100.0
- strict-gloss-query-bm25-chunkmax: bm25 R@2 3.85, chunk-max R@2 23.08, union R@2 26.28, both-miss slots 115, queries blind at 2: 46, full-list RRF R@2 12.18, RRF-on-union R@2 12.82
- strict-docs-gloss-bm25-chunkmax: bm25 R@2 3.21, chunk-max R@2 21.15, union R@2 24.36, both-miss slots 118, queries blind at 2: 49, full-list RRF R@2 10.26, RRF-on-union R@2 12.18

## Drowning (LIMIT-small golds, distractors from data/limit)

| added | whole R@2 | whole semantic-only R@2 | MaxP R@2 | MaxP semantic-only R@2 | MaxP exact-string-only R@2 | BM25 R@2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 16.2 | 16.2 | 98.15 | 98.15 | 98.15 | 99.15 |
| 50 | 9.7 | 10.4 | 75.3 | 97.35 | 75.75 | 79.05 |
| 100 | 6.55 | 7.3 | 66.5 | 96.7 | 67.3 | 72.3 |
| 200 | 4.8 | 5.65 | 60.2 | 95.1 | 61.95 | 64.25 |
| 500 | 2.85 | 3.55 | 54.65 | 93.75 | 57.1 | 57.25 |
| 1000 | 2.05 | 2.65 | 51.8 | 91.9 | 55.4 | 55.4 |
| 2000 | 1.4 | 1.8 | 50.5 | 90.6 | 54.75 | 54.1 |
| 5000 | 0.75 | 0.95 | 46.5 | 87.4 | 52.05 | 50.55 |
| 10000 | 0.4 | 0.55 | 41.15 | 83.35 | 48.2 | 47.1 |
| 20000 | 0.15 | 0.15 | 37.3 | 78.95 | 45.7 | 45.2 |
| 50000 | 0.0 | 0.0 | 21.85 | 71.95 | 25.95 | 28.6 |

