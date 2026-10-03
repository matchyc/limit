# LIMIT: reproduced facts and research directions

Four independent checks are in `experiments/`. Recall numbers below are on a 0–100 scale. The moment-curve separation was recomputed outside the subagent script: 1000/1000 queries are strict, and all 1035 pairs have integer margin at least 1.

`data/limit` and `data/limit-small` are different draws. Document-id overlap is 0. All 1000 shared query ids have different text (example: small `query_0` is “Who likes Joshua Trees?”; full `query_0` is “Who likes Birch Trees?”). Only 552 query strings coincide. Do not treat the small corpus as a subset of the 50k corpus.

## What reproduced

Lexical overlap explains BM25, not a high-dimensional semantic match. The queried attribute string is inside both relevant documents for 1000/1000 queries on both files. BM25 (`k1=1.5`, `b=0.75`) scores:

| Corpus | Method | R@2 | R@10 | R@20 or R@100 |
| --- | --- | ---: | ---: | ---: |
| limit-small | BM25 | 99.15 | 100 | 100 (R@20) |
| limit-small | paper BM25 | 97.8 | 100 | 100 |
| limit | BM25 | 92.65 | 94.60 | 96.30 (R@100) |
| limit | paper BM25 | 85.7 | 90.4 | 93.6 |

Full-corpus BM25 is higher than the paper, so this is the same qualitative result under a different tokenizer, not a digit-level reproduction. Substring search is even higher (small 99.5, full 96.35 at R@2). On the full file, 936/1000 attributes occur in no irrelevant document. The failures are string collisions such as `Go` inside `Gorges`.

Exact top-2 geometry exists in dimension 4. Documents on `(t, t^2, t^3, t^4)` with `t = i+1`, and the query equal to the non-constant part of `-((x-t_a)(x-t_b))^2`, put both gold documents strictly above the other 44 for every LIMIT-small query. Minimum integer margin is 1 (median 169). The same separator works for all 1035 pairs. Dimension 3 is not enough for this curve: the degree-3 LP is strict on 118/1000 queries. After unit-length normalization the same perfect queries have a minimum margin of about `4.6e-6` and a median of about `0.0056`.

InfoNCE does not search for that geometry. A short run (Adam, lr 0.01, temperature 0.1, unit norm, at most 3000 steps; the paper uses 100000) gets perfect Recall@2 for `d=4` at `n=6` and `n=8`, and fails at the paper’s critical point `n=10` (best Recall@2 94.4). For `d=8` it is perfect through `n=24` (paper critical-n is 28; `n=28` was not run). Freezing documents on the moment curve and training only queries with InfoNCE reaches Recall@2 50.4 at `n=16` (120 pairs) and 18.0 at `n=46` (200 pairs). Closed-form queries on those same documents score Recall@2 100, but their InfoNCE loss is worse: 2.199 vs Adam’s 1.896 at `n=16`, and 3.248 vs 2.910 at `n=46`.

A small embedder fails in the same direction as the paper’s large ones, and faster. `all-MiniLM-L6-v2` on LIMIT-small: Recall@2 16.2, Recall@10 48.05, Recall@20 69.45. A content-word overlap ranker on the same 46 documents scores Recall@2 97.2. For “Who likes Chairs?”, the gold documents contain the word and still rank 24 and 36; the top document shares no content word with the attribute. Adding documents from the other corpus (not the official negatives for these queries) drops Recall@2 from 16.2 (46 docs) to 6.55 (146), 2.05 (1046), 0.40 (10046), and 0.00 (50046).

## Directions

**1. The training objective, not the sign-rank, is the free-embedding bottleneck.**

Question: for fixed document geometry that is already separable, which loss has its optimum at a perfect top-2 ranking? InfoNCE with temperature 0.1 prefers a lower loss at Recall@2 50 than at the perfect closed form, because unit-norm margins are far below the temperature. Compare full-batch InfoNCE, a pairwise hinge on the gold-vs-best-negative gap, and a listwise top-2 surrogate, on the same moment-curve documents. The prediction is that hinge reaches Recall@2 100 in dimension 4 for `n=46`, while InfoNCE does not, at the same step budget.

**2. Exact capacity is cheap; a margin that survives normalization is not.**

Question: what is the smallest dimension in which all `C(46,2)` pair queries have inner-product margin at least `ε` after unit-norm constraints on both sides? Exact separation is done at `d=4`, but the unit-norm minimum margin is about `4.6e-6`, which will not survive int8 quantization or an approximate nearest-neighbor graph. Sweep `ε` in `{1e-3, 1e-2, 0.05}` and `d` upward. This is the robust version of LIMIT, and it is the version that can actually constrain an index.

**3. Remove the lexical shortcut before blaming the vector space.**

Question: if each attribute is paraphrased in the documents and the query never shares a rare token with them, does BM25 collapse while a model that pools per attribute still solves the same qrel graph? Today the rarest attribute token has document frequency 2 for most queries, so LIMIT does not separate “the string is visible” from “the combination is representable.” Build that paraphrase split from the existing qrels. Keep the 46-person graph fixed so the only change is surface form.

**4. Single-attribute binding inside a long likes-list.**

Question: at what list length does a mean-pooled encoder drop a gold attribute below the best negative, when the lexical overlap ranker is still perfect? MiniLM already fails at the natural list length of this dataset, and its top errors are not near-synonyms of the attribute. Measure gold rank versus number of filler attributes, then replace mean pooling with max-over-spans or a late-interaction score on the same backbone. The useful outcome is a curve, not another single Recall@2.
