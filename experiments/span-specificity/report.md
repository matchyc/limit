# Span specificity on frozen MiniLM

Model `sentence-transformers/all-MiniLM-L6-v2` on cuda. LIMIT-small: 46 docs, 1000 queries, 2070 attribute chunks, 126 gloss queries. 7.5s.

One score. Chunk-max over a phrasing, optionally the max over six phrasings, then divide by a document frequency. Score-band df counts documents whose chunk-max lies within ±δ (including itself). Span df counts documents that have a chunk with cosine ≥ τ to the winning chunk.

Recall is macro over the two golds, in percent.

## Original wording (control)

| score | R@2 | R@10 | R@20 | both@2 | both@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| whole-doc | 16.2 | 48.05 | 69.45 | 2.3 | 23.2 |
| whole-doc-max-template | 16.6 | 47.75 | 68.25 | 2.8 | 24.3 |
| bm25 | 99.15 | 100.0 | 100.0 | 98.4 | 100.0 |
| bm25-max-template | 98.25 | 100.0 | 100.0 | 97.0 | 100.0 |
| chunk-max | 98.15 | 100.0 | 100.0 | 96.4 | 100.0 |
| chunk-max-max-template | 97.85 | 100.0 | 100.0 | 95.7 | 100.0 |

### Score-band divisor on chunk-max

| δ | R@2 | R@10 | both@2 |
| ---: | ---: | ---: | ---: |
| 0.00 | 98.15 | 100.0 | 96.4 |
| 0.01 | 86.7 | 99.75 | 81.2 |
| 0.02 | 87.15 | 99.9 | 80.0 |
| 0.05 | 93.75 | 99.9 | 88.3 |
| 0.10 | 97.25 | 100.0 | 94.8 |

### Span divisor on chunk-max

| τ | R@2 | R@10 | both@2 |
| ---: | ---: | ---: | ---: |
| 0.50 | 37.8 | 74.95 | 17.2 |
| 0.70 | 94.3 | 96.35 | 92.6 |
| 0.80 | 98.15 | 100.0 | 96.4 |
| 0.90 | 98.15 | 100.0 | 96.4 |
| 0.95 | 98.15 | 100.0 | 96.4 |

## Gloss queries

| score | R@2 | R@10 | R@20 | both@2 | both@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| whole-doc | 7.94 | 31.75 | 57.54 | 0.0 | 13.49 |
| whole-doc-max-template | 7.94 | 30.16 | 52.38 | 0.0 | 8.73 |
| bm25 | 3.57 | 24.21 | 47.22 | 0.0 | 7.14 |
| bm25-max-template | 3.97 | 24.6 | 45.63 | 0.0 | 7.14 |
| chunk-max | 24.21 | 62.7 | 77.38 | 11.11 | 50.0 |
| chunk-max-max-template | 23.41 | 59.92 | 77.38 | 11.9 | 45.24 |

Per phrasing, chunk-max:

- who-likes: R@2 24.21 | R@10 62.7 | R@20 77.38 | both@2 11.11 | both@10 50.0
- who-enjoys: R@2 25.79 | R@10 61.51 | R@20 76.19 | both@2 11.9 | both@10 48.41
- person-enjoys: R@2 24.6 | R@10 58.33 | R@20 77.78 | both@2 11.11 | both@10 42.86
- someone-likes: R@2 23.81 | R@10 59.92 | R@20 78.97 | both@2 11.9 | both@10 44.44
- find-person: R@2 24.21 | R@10 59.52 | R@20 76.98 | both@2 10.32 | both@10 46.83
- which-person: R@2 24.6 | R@10 59.92 | R@20 77.78 | both@2 11.9 | both@10 46.03

### Score-band on gloss chunk-max

| δ | single R@2 | single R@10 | max-template R@2 | max-template R@10 |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 24.21 | 62.3 | 23.41 | 59.92 |
| 0.01 | 21.03 | 54.37 | 21.03 | 51.59 |
| 0.02 | 21.43 | 60.32 | 19.44 | 54.37 |
| 0.05 | 22.22 | 59.52 | 21.83 | 56.75 |
| 0.10 | 24.21 | 61.51 | 21.83 | 59.13 |

### Span divisor on gloss chunk-max

| τ | single R@2 | single R@10 | max-template R@2 | max-template R@10 |
| ---: | ---: | ---: | ---: | ---: |
| 0.50 | 12.3 | 39.68 | 13.49 | 37.3 |
| 0.70 | 23.02 | 62.7 | 22.22 | 59.92 |
| 0.80 | 24.21 | 62.7 | 23.41 | 59.92 |
| 0.90 | 24.21 | 62.7 | 23.41 | 59.92 |
| 0.95 | 24.21 | 62.7 | 23.41 | 59.92 |

## Planted exact copies

Each query is scored against the 46 documents plus t other people whose only sentence is `Copy Person i likes {attribute}.` Gap is the worse gold's chunk score minus the best copy.

### original-who-likes

| t | chunk R@2 | chunk R@10 | whole R@10 | bm25 R@2 | bm25 R@10 | median gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t=1 | 58.65 | 100.0 | 45.2 | 49.7 | 100.0 | -0.0392 |
| t=2 | 26.7 | 100.0 | 42.25 | 0.0 | 100.0 | -0.0692 |
| t=5 | 1.9 | 99.95 | 31.0 | 0.0 | 100.0 | -0.1137 |

With t=5 the corpus has 51 documents, so R@10 still counts a gold that sits just behind the copies. R@2 is the cut that shows whether those copies took the top.

Score-band R@2:

| δ | t=1 | t=2 | t=5 |
| ---: | ---: | ---: | ---: |
| 0.00 | 58.65 | 26.7 | 1.9 |
| 0.01 | 47.8 | 28.4 | 12.55 |
| 0.02 | 47.85 | 33.15 | 27.5 |
| 0.05 | 54.7 | 37.75 | 40.75 |
| 0.10 | 58.25 | 31.65 | 21.75 |

Span R@2:

| τ | t=1 | t=2 | t=5 |
| ---: | ---: | ---: | ---: |
| 0.50 | 13.8 | 10.0 | 6.4 |
| 0.70 | 55.95 | 94.3 | 93.6 |
| 0.80 | 58.65 | 98.15 | 98.15 |
| 0.90 | 58.65 | 98.15 | 98.1 |
| 0.95 | 58.65 | 42.85 | 45.4 |

### original-max-template

| t | chunk R@2 | chunk R@10 | whole R@10 | bm25 R@2 | bm25 R@10 | median gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t=1 | 55.1 | 100.0 | 45.05 | 49.7 | 100.0 | -0.0544 |
| t=2 | 19.65 | 100.0 | 41.95 | 0.0 | 100.0 | -0.0823 |
| t=5 | 1.3 | 99.9 | 30.85 | 0.0 | 100.0 | -0.1267 |

With t=5 the corpus has 51 documents, so R@10 still counts a gold that sits just behind the copies. R@2 is the cut that shows whether those copies took the top.

Score-band R@2:

| δ | t=1 | t=2 | t=5 |
| ---: | ---: | ---: | ---: |
| 0.00 | 55.1 | 19.65 | 1.3 |
| 0.01 | 46.6 | 24.45 | 12.0 |
| 0.02 | 46.1 | 33.05 | 31.15 |
| 0.05 | 52.65 | 37.25 | 46.75 |
| 0.10 | 54.65 | 27.7 | 29.85 |

Span R@2:

| τ | t=1 | t=2 | t=5 |
| ---: | ---: | ---: | ---: |
| 0.50 | 11.3 | 8.75 | 5.5 |
| 0.70 | 52.7 | 94.0 | 93.3 |
| 0.80 | 55.1 | 97.85 | 97.85 |
| 0.90 | 55.1 | 97.85 | 97.8 |
| 0.95 | 55.1 | 37.4 | 41.95 |

BM25 given the same six phrasings, then max: t=1 R@10 100.0; t=2 R@10 99.9; t=5 R@10 48.3


### gloss-who-likes

| t | chunk R@2 | chunk R@10 | whole R@10 | bm25 R@2 | bm25 R@10 | median gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t=1 | 16.67 | 61.11 | 29.76 | 3.17 | 23.41 | -0.0173 |
| t=2 | 11.9 | 60.71 | 27.38 | 2.78 | 21.43 | -0.0355 |
| t=5 | 7.94 | 53.57 | 18.65 | 2.78 | 17.46 | -0.0571 |

With t=5 the corpus has 51 documents, so R@10 still counts a gold that sits just behind the copies. R@2 is the cut that shows whether those copies took the top.

Score-band R@2:

| δ | t=1 | t=2 | t=5 |
| ---: | ---: | ---: | ---: |
| 0.00 | 16.67 | 11.9 | 7.94 |
| 0.01 | 15.87 | 14.29 | 13.49 |
| 0.02 | 16.27 | 14.29 | 12.3 |
| 0.05 | 15.08 | 12.3 | 11.11 |
| 0.10 | 16.27 | 11.9 | 7.94 |

Span R@2:

| τ | t=1 | t=2 | t=5 |
| ---: | ---: | ---: | ---: |
| 0.50 | 6.75 | 7.54 | 4.76 |
| 0.70 | 15.87 | 23.02 | 23.02 |
| 0.80 | 16.67 | 24.21 | 24.21 |
| 0.90 | 16.67 | 24.21 | 24.21 |
| 0.95 | 16.67 | 15.08 | 14.68 |

### gloss-max-template

| t | chunk R@2 | chunk R@10 | whole R@10 | bm25 R@2 | bm25 R@10 | median gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t=1 | 18.25 | 59.92 | 27.38 | 3.17 | 23.41 | -0.0298 |
| t=2 | 10.32 | 57.94 | 25.0 | 2.78 | 21.43 | -0.0482 |
| t=5 | 5.95 | 51.59 | 17.86 | 2.78 | 17.46 | -0.065 |

With t=5 the corpus has 51 documents, so R@10 still counts a gold that sits just behind the copies. R@2 is the cut that shows whether those copies took the top.

Score-band R@2:

| δ | t=1 | t=2 | t=5 |
| ---: | ---: | ---: | ---: |
| 0.00 | 18.25 | 10.32 | 5.95 |
| 0.01 | 14.68 | 10.71 | 8.33 |
| 0.02 | 14.29 | 11.51 | 10.71 |
| 0.05 | 18.25 | 11.9 | 11.51 |
| 0.10 | 17.86 | 10.71 | 5.56 |

Span R@2:

| τ | t=1 | t=2 | t=5 |
| ---: | ---: | ---: | ---: |
| 0.50 | 6.35 | 7.54 | 5.56 |
| 0.70 | 17.06 | 22.22 | 22.22 |
| 0.80 | 18.25 | 23.41 | 23.41 |
| 0.90 | 18.25 | 23.41 | 23.41 |
| 0.95 | 18.25 | 12.7 | 14.29 |

BM25 given the same six phrasings, then max: t=1 R@10 21.83; t=2 R@10 19.84; t=5 R@10 12.3


## Diagnostics

- chunk-cosine-offdiag-sample: n=200000 mean 0.25 median 0.2422 p10 0.142 p90 0.3538
- gold-winning-chunk-vs-best-copy: n=1000 mean 0.5965 median 0.6019 p10 0.5204 p90 0.6682
- two-golds-winning-chunks: n=1000 mean 0.5827 median 0.5888 p10 0.4859 p90 0.6706
- two-copies: n=1000 mean 0.9442 median 0.9447 p10 0.9349 p90 0.953
- score-gap-worse-gold-minus-best-copy: n=1000 mean -0.1166 median -0.1137 p10 -0.1685 p90 -0.0675
- top1-chunk-contains-attribute: {'hits': 994, 'queries': 1000}

