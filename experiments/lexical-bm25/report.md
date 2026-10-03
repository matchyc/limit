# LIMIT lexical BM25

## Command

```bash
/home/yangshen/miniconda3/bin/python3 /mnt/raid0nvme0/yangshen/meng/vector_search/limit/experiments/lexical-bm25/run_lexical_bm25.py
```

Python 3.13.11 (Anaconda), numpy 2.4.3. Standard library plus numpy only. The script writes `results.json` in this directory.

## Setup

Attribute string: the query text with the prefix `Who likes ` and the trailing `?` removed, case preserved. Occurrence is a case-sensitive substring test on document `text` (titles are empty).

BM25 uses k1=1.5, b=0.75, and Lucene IDF `ln(1 + (N - df + 0.5) / (df + 0.5))`. Tokens are lowercase maximal `[a-z0-9]+` runs, so whitespace and punctuation are separators (`Rubik's` → `rubik`, `s`). Each query term type is scored once. Document frequency and average length are fit on the corpus being ranked. Ties break by score descending, then document `_id` ascending. The substring baseline uses the same tie break with score 1 when the raw attribute is a substring and 0 otherwise.

Recall is the macro-average over 1000 queries of `|top-k ∩ relevant| / 2`, reported on the paper's 0–100 scale. Every query has exactly two relevant documents.

## These two files are separate draws

`limit-small` and `limit` share query ids and both have 1000 queries, 2 relevant documents per query, and 46 relevant people. They do not share query text or people: 552 attribute strings overlap, 0 relevant people overlap, and 0 of the 46 small-corpus ids appear in the 50k corpus. Each split is scored with its own queries, corpus, and qrels. The usual LIMIT construction, in which the 46-document corpus is the relevant subset of the same queries, is not what these files contain.

## Qrel graph

Computed separately on each qrel file. The two graphs have the same shape and disjoint vertex sets.

| | value |
| --- | --- |
| nodes | 46 |
| edges | 1000 |
| edges with more than one query | 0 |
| possible edges `C(46, 2)` | 1035 |
| density `1000/1035` | 0.9662 |
| mean queries per person | 43.478 (2000/46) |
| queries per person | min 37, max 45 |

Mean queries per person equals mean number of unique partners, because each pair is used by exactly one query.

## Attribute leaks

On both corpora the attribute string occurs in both relevant documents for **1000/1000** queries. Case-folding changes nothing.

| corpus | queries with the attribute in any irrelevant doc | irrelevant-doc count |
| --- | --- | --- |
| limit-small | **0.011** (11/1000) | min 0, median 0, mean 0.028, max 6. Histogram: 0→989, 1→1, 2→7, 3→1, 4→1, 6→1 |
| limit | **0.064** (64/1000) | min 0, median 0, mean 51.823, max 18122. 936 queries have count 0. Ten queries exceed 1000 (`Go` 18122, `Snow` 9663, `Iron` 5148, `Forests` 2707, `Apple Cider` 2675, `Honey` 2672, `Venus` 2668, `Horses` 2613, `Clouds` 2600, `Cricket` 2576) |

The exact substring set equals the relevant pair for 989/1000 small queries and 936/1000 full queries. The extra hits are prefix collisions inside longer phrases or names (`Ham` inside `Hamburgers`, `Horses` inside `Horseshoe Crabs`, `Go` inside `Gorges`, `Sage` inside a first name), not a second copy of the full like-item in the small corpus.

## Recall

| corpus | method | R@2 | R@10 | R@20 | R@100 |
| --- | --- | ---: | ---: | ---: | ---: |
| limit-small | BM25 | 99.15 | 100 | 100 | — |
| limit-small | substring | 99.50 | 100 | 100 | — |
| limit-small | paper BM25 | 97.8 | 100 | 100 | — |
| limit | BM25 | 92.65 | 94.60 | — | 96.30 |
| limit | substring | 96.35 | 98.30 | — | 99.05 |
| limit | paper BM25 | 85.7 | 90.4 | — | 93.6 |

Small BM25 hit counts at 2: 984 queries retrieve both relevant docs, 15 retrieve one, 1 retrieves none. Full BM25: 910 both, 33 one, 57 none. At 100 on the full corpus, 952 queries still retrieve both, 22 retrieve one, and 26 retrieve none.

## Paper comparison

Small BM25 matches the paper within a few points: Recall@2 is 1.35 points higher (99.15 vs 97.8), and Recall@10 and Recall@20 match at 100.

Full BM25 is higher than the paper by 6.95, 4.20, and 2.70 points (92.65/94.60/96.30 vs 85.7/90.4/93.6). Recall@100 is within a few points; Recall@2 is not. The result is the same kind of outcome — BM25 recovers the large majority of relevant documents — on a different 46-person draw, with this tokenizer and Lucene IDF.

## Why BM25 succeeds

BM25 succeeds because the attribute is written verbatim in both relevant documents, and the rarest token of that attribute is usually private to those two documents.

Measured support:

- Relevant documents are missing an attribute token in **0/1000** queries on both corpora.
- A document that lacks the rarest attribute token never outranks a relevant document (**0/1000** on both corpora).
- That rarest token has document frequency 2 for **936/1000** small queries and **841/1000** full queries. Those two carriers are then the entire top 2.

The remaining errors are queries whose rarest token is shared. On the full corpus, single-token queries score Recall@2 **96.85** (571 queries) and multi-token queries score **87.06** (429 queries). Multi-word attributes are split, and filler documents recombine the pieces (`hot` df 9854 with `chocolate` df 5164; `black` df 14076 with `tea` df 5045). Length normalization then ranks a shorter carrier of those tokens above a relevant person. The substring baseline keeps the contiguous phrase, so its full-corpus Recall@2 is higher (96.35 vs 92.65), except where the raw string is a prefix of another capitalized phrase.
