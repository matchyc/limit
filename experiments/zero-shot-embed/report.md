# Zero-shot MiniLM on LIMIT

## Setup

Model `sentence-transformers/all-MiniLM-L6-v2` (384-d), cosine similarity, no query prefix. Device: CUDA_VISIBLE_DEVICES=1, seen as cuda:0, NVIDIA RTX PRO 6000 Blackwell Server Edition (capability (12, 0)). Torch 2.10.0+cu128, sentence-transformers 6.1.0.

Wall time 21.7s (model load 1.8s, embed 1000 queries 0.2s, embed 46 docs 0.0s, embed 50000 docs 16.8s, scoring 0.9s).

Relevance is the limit-small qrels: 1000 queries, exactly 2 gold documents each, all inside the 46-document pool. Recall@k is the macro-average of |gold ∩ top-k| / 2, reported on a 0–100 scale. Equal cosines break toward the alphabetically earlier document id. TF32 was disabled for the similarity product.

MiniLM truncates at 256 tokens. On the 46-document pool, token length min/median/max = 143/161/173, truncated 0/46. On every 25th full-corpus document, truncated 0/2000 (max tokens 184).

## limit-small (46 documents)

| k | Recall@k |
| ---: | ---: |
| 2 | 16.20 |
| 10 | 48.05 |
| 20 | 69.45 |

## Drowning curve

The 46 relevant-pool documents stay in the index. Added documents are a nested prefix of one `numpy.random.default_rng(0).permutation` of the 50,000 full-corpus rows (seed 0). The last point ranks all 50,046 documents exactly; no approximate search.

| Added | Total docs | Recall@2 | Recall@10 |
| ---: | ---: | ---: | ---: |
| 0 | 46 | 16.20 | 48.05 |
| 100 | 146 | 6.55 | 22.20 |
| 1000 | 1046 | 2.05 | 6.05 |
| 10000 | 10046 | 0.40 | 1.65 |
| 50000 | 50046 | 0.00 | 0.40 |

## Comparison with the paper's large embedders

On the 46-document pool MiniLM scores Recall@2 = 16.20, Recall@10 = 48.05, Recall@20 = 69.45. A ranker that only counts shared attribute content words, with the same id tie-break, scores Recall@2 = 97.20, Recall@10 = 99.30, Recall@20 = 99.60 on these same qrels (both golds in the top 2 for 95.9% of queries). That sits next to the paper's BM25 figure of about 98. A 4096-d embedder (Promptriever) is about 54. MiniLM is in the same direction as those dense models, and further from lexical retrieval: on a 46-document corpus the two gold documents are easy for word overlap and still usually missed by this embedder. Random Recall@2 here would be 2/46 = 4.35, so 16.20 is above chance and still a failure.

Adding irrelevant-pool documents drops MiniLM from Recall@2 = 16.20 at 46 documents to Recall@2 = 0.00 and Recall@10 = 0.40 at 50046 documents. The paper's dense models on the official 50k corpus are about Recall@2 = 3 and Recall@100 = 19, while BM25 stays near 86. This curve is not that official qrel file: the 50k corpus here uses different document ids, and its qrels do not mark the same 46 people. It is the drowning test that was asked for, and it moves the same way: extra documents collapse embedding recall while the labeled golds stay fixed.

## Error analysis

Sample: 200 queries from `numpy.random.default_rng(0).choice` (then sorted), ranked inside the 46-document pool. A content word is an alphanumeric token longer than 3 characters. The attribute is the span inside `Who likes ...?`. The top wrong document is the highest-ranked document that is not one of the two golds.

On this sample, Recall@2 = 17.00 and Recall@10 = 49.75. Top-1 is a gold document for 16.0% of queries. Both golds are in the top 2 for 2.0%. The better gold has mean rank 7.30 (median 4.5); the worse gold has mean rank 20.77 (median 18.5).

Better-gold rank histogram (of 200): 1=32, 2=32, 3-5=48, 6-10=37, 11-20=38, 21+=13. Worse-gold rank histogram: 1=0, 2=4, 3-5=16, 6-10=30, 11-20=55, 21+=95.

The top wrong document shares an attribute content word on only 9.5% of the 200 queries (mean 0.10 shared words). A typical non-gold document in the 46 shares a content word 5.1% of the time, so lexical overlap is only mildly enriched. Among the 168 queries whose top-1 is already wrong, 10.1% have that overlap. The usual mistake is a biography that does not repeat the attribute.

Dataset check: 1990/1990 gold documents whose attribute has a content word contain every such word (100.0%). Attributes with no alphanumeric token longer than 3 characters are: Ham, Pop Art, Yo-yos, Law, Uno. Across the 995 queries whose attribute has a content word, a non-gold document in the 46 shares such a word 4.5% of the time. The gold text states the attribute, word overlap recovers it (Recall@2 = 97.20), and MiniLM still does not put both golds first. The embedding is not tracking the queried attribute inside an otherwise similar biography.

The same 200 queries on the 50,046-document pool: Recall@2 = 0.00, Recall@10 = 0.25, mean rank of the better gold = 7301.5 (median 4499.5). Top-1 is wrong on 200/200 queries, and the top wrong document shares an attribute content word on 39.0% of queries (39.0% of the top-1 errors).

Examples where top-1 is not gold (limit-small ranks):

Top wrong document shares an attribute content word:

- `query_4` attribute `Disco Music`: gold ranks Shelvia Goike@7 (cosine 0.228), Geneva Durben@13 (cosine 0.214); top wrong `Riya Hayhoe` at rank 1 (cosine 0.258) sharing ['music'].
- `query_13` attribute `Soy Sauce`: gold ranks Geneva Durben@14 (cosine 0.197), Jerrod Dumpit@27 (cosine 0.145); top wrong `Riya Hayhoe` at rank 1 (cosine 0.311) sharing ['sauce'].
- `query_14` attribute `Elm Trees`: gold ranks Amaris Grow@2 (cosine 0.397), Geneva Durben@17 (cosine 0.301); top wrong `Theola Laudermilk` at rank 1 (cosine 0.406) sharing ['trees'].

Top wrong document shares no attribute content word:

- `query_7` attribute `Chairs`: gold ranks Geneva Durben@24 (cosine 0.160), Gladstone Oonk@36 (cosine 0.135); top wrong `Armand Schweda` at rank 1 (cosine 0.224).
- `query_15` attribute `Barley`: gold ranks Geneva Durben@13 (cosine 0.262), Marcellus Meachum@29 (cosine 0.196); top wrong `Nathaniel Robens` at rank 1 (cosine 0.372).
- `query_18` attribute `Limes`: gold ranks Laurine Bellizzi@17 (cosine 0.280), Geneva Durben@35 (cosine 0.215); top wrong `Georgette Cagna` at rank 1 (cosine 0.350).

Per-query ranks for all 200 sampled queries are in `results.json`.
