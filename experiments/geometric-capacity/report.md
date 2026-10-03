# LIMIT-small geometric capacity

Constructive check of the claim that top-2 inner-product retrieval on LIMIT-small fits in dimension 4. Documents are fixed. A query vector is accepted when both gold documents strictly outrank the other 44 under an inner product. Vectors are not L2-normalized, so this is not a cosine result.

Corpus order is appearance order in `data/limit-small/corpus.jsonl` (46 ids). Document i, starting at 0, is placed at `t = i+1` on `(t, t^2, t^3, t^4)`. Each of the 1000 queries has two gold documents. There are `C(46, 2) = 1035` possible pairs; the qrels use 1000 of them.

## Degree-4 moment curve

The separator is the polynomial `p(x) = -((x - t_a)(x - t_b))^2`. It is zero at the two gold parameters and strictly negative at every other real number. It has degree 4, so

`p(x) = c + q1 x + q2 x^2 + q3 x^3 + q4 x^4`.

The constant `c` is the same for every document and does not change inner-product order. The query is `(q1, q2, q3, q4)` in the monomial coordinates. On integer nodes this margin is a positive integer. Python integers give a minimum margin of **1** over all 1035 pairs (median **169.0**). Float64 dots reproduce those integers.

| method | Recall@2 | Recall@10 | both-in-top-2 | strict success | median margin | min margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| moment_d4_integer_closed_form | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 169 | 1 |
| moment_d4_integer_lp | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.12453e-05 | 1.63715e-07 |
| moment_d4_chebyshev_poly_eval | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.73421e-05 | 1.17317e-10 |
| moment_d4_chebyshev_monomial_dot | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.73421e-05 | 1.17317e-10 |

Closed form, integer nodes, float64 inner product: **1000/1000** queries are strict top-2 successes. Recall@2 = **1.0000**, Recall@10 = **1.0000**, both-in-top-2 rate = **1.0000**. Among those successes the positive-vs-best-negative margin has median **169.0** and minimum **1.0**. The largest absolute gap between the float64 margin and the Python-int margin is **0.0**. Unit-L2 query margins (raw margin divided by `||q||_2`) have median **0.005573044473045911** and minimum **4.555662843639353e-06**.

The same closed form on all 1035 document pairs, not only the 1000 qrels, is strict on **1035/1035** pairs. Float64 discrepancy versus integers: **0.0**.

HiGHS, on the same integer curve after dividing each coordinate by its max absolute value, finds a strictly positive margin for **1000/1000** queries (Recall@2 **1.0000**, both-in-top-2 **1.0000**). Median / minimum recomputed margins inside that unit box: **3.12453e-05** / **1.63715e-07**. 286 of the 1000 margins are at most 1e-5. That is the price of `||q||_inf <= 1` on a Vandermonde basis, not a failure to rank: every query is still strict, and the unnormalized integer polynomial keeps a margin of at least 1.

Chebyshev nodes in (0, 1), same polynomial as a product: both-in-top-2 **1.0000**, strict (margin > 0) **1.0000**, median margin **1.73421e-05**, minimum **1.17317e-10**. 11 of those positive margins are below 1e-8 and 427 are below 1e-5. They are small because `t` is in (0, 1), not because a third document ties or wins. The float64 monomial dot matches this ranking: minimum margin **1.17317e-10**, centered-score gap versus the product **8.88178e-16**.

## Where dimension actually binds

Degree 3, features `(t, t^2, t^3)`, same integer nodes, same LP: strict success **0.1180** (118/1000), Recall@2 **0.1390**, Recall@10 **0.3030**, both-in-top-2 **0.1180**. Successful-query margins median / min **0.000226447** / **1.54105e-05**. The misses are real geometric failures. A degree-3 moment curve is only 1-neighborly, so not every pair is an exposed face. Failures have non-positive margin.

Degree 2, features `(t, t^2)`: strict success **0.0390** (39/1000), Recall@2 **0.0820**, Recall@10 **0.2540**. The misses are real geometric failures. A degree-2 moment curve is only 1-neighborly, so not every pair is an exposed face. Failures have non-positive margin.

A cyclic polytope of dimension d is floor(d/2)-neighborly. Every pair of vertices is a face once d >= 4, and not before. The degree-2 and degree-3 rows are the matching negative controls. The constant monomial, which would make a 5-dimensional homogeneous model, is not required for inner-product comparisons.

## Adam on the hinge

The hinge objective was `||q||^2 + C * sum relu(m - (s_pos - s_neg))` over the 2 positives and 44 negatives, 800 Adam steps, seed 0.

| run | Recall@2 | both-in-top-2 | strict success | median success margin | min success margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| moment_d4_adam_maxnorm | 0.1160 | 0.0140 | 0.0140 | 0.00453311 | 0.00057144 |
| moment_d4_adam_raw | 0.1245 | 0.0360 | 0.0360 | 18.6175 | 0.717792 |
| chebyshev_basis_adam | 0.3770 | 0.1630 | 0.1630 | 0.00823283 | 2.23694e-05 |

Max-normalized monomials (`C=10`, target margin 1, lr 0.05): strict success **0.0140**. Raw monomials: **0.0360**. Chebyshev polynomial basis (`C=20`, target margin 0.05, lr 0.02): **0.1630**. These misses are optimization failures. The closed form already separates every one of these queries by an integer margin of at least 1, and the LP recovers a positive margin on the same degree-4 point set. Sum-of-hinges Adam spends its step budget on the average violated pair and on the large monomial coordinates; it does not certify infeasibility.

## Gaussian controls

Document matrices are i.i.d. standard normal. Each query is an independent HiGHS max-margin LP in that fixed cloud, with `||q||_inf <= 1`. A pair is strictly realizable only if it is an exposed edge of the convex hull.

| embedding | Recall@2 | Recall@10 | both-in-top-2 | strict success | median success margin | min success margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gaussian_d4_seed0 | 0.2000 | 0.3645 | 0.1570 | 0.1560 | 0.218344 | 5.73757e-05 |
| gaussian_d4_seed1 | 0.1715 | 0.3215 | 0.1320 | 0.1310 | 0.220254 | 0.00105372 |
| gaussian_d4_seed2 | 0.1770 | 0.3420 | 0.1460 | 0.1450 | 0.285376 | 0.00239588 |
| gaussian_d8_seed0 | 0.7900 | 0.8370 | 0.7750 | 0.7740 | 0.857913 | 0.000400137 |
| gaussian_d8_seed1 | 0.7340 | 0.7765 | 0.7110 | 0.7100 | 0.677531 | 0.00110923 |
| gaussian_d8_seed2 | 0.7535 | 0.8020 | 0.7300 | 0.7290 | 0.918544 | 0.000692098 |

Best control: **gaussian_d8_seed0**. Recall@2 **0.7900**, Recall@10 **0.8370**, both-in-top-2 **0.7750**, strict success **0.7740** (774/1000). Median / minimum margin among strict successes: **0.857913** / **0.000400137**. The misses are real: those pairs are not exposed edges of this cloud (226 queries with margin <= 1e-12). 0 further queries have a tiny positive margin <= 1e-05 and are not counted as strict.

A random cloud in dimension 4 realizes 13–16% of the gold pairs. The best dimension-8 cloud realizes 77.4%. Neither matches the moment curve, which realizes every pair in dimension 4. The comparison uses the same per-query max-margin LP; only the document geometry changes.

## Lexical ceiling

One coordinate per query attribute (dimension 1000). The query is a one-hot, so the score of a document is 1 when the attribute is present and 0 otherwise.

Substring test, as specified (`attribute in document text`): Recall@2 **0.9935**, Recall@10 **1.0000**, both-in-top-2 **0.9900**, strict success **0.9890**. Support equals the gold pair for **989** queries; **11** queries also match an extra document; **0** miss a gold document. 989 queries have attribute support equal to the gold pair (strict margin 1); 11 queries tie extra documents at the same score, so the margin is 0. Those are tie failures of a binary lexical score, not near-miss numerical margins.

The extra hits are shorter attributes occurring inside longer list items (for example `Ham`, `Horses`, `Jelly`, `Rings`, `Barbers`, `Venus`, `Cucumbers`, `Cleaners`). Tied documents all score 1, so the positive-vs-best-negative margin is 0. With a stable lower-index tie break, both gold documents are not always the two that occupy the top-2 slots.

Parsed likes-list items, exact string match against the attribute: Recall@2 **1.0000**, Recall@10 **1.0000**, both-in-top-2 **1.0000**, strict success **1.0000**, support equals gold on **1000/1000** queries. This is the lexical ceiling of the task as generated: membership of a list item, not a semantic paraphrase.

## What this says about dimension

Dimension 4 does exactly solve LIMIT-small for unnormalized inner products. The integer moment curve plus the degree-4 query polynomial puts both gold documents strictly above the other 44 on **1000/1000** queries, with Recall@2 = Recall@10 = **1.0000** and minimum margin **1.0**. The same statement holds for all **1035/1035** document pairs. Degree 3 solves only **118/1000** of these queries, so the pattern is not realizable in every dimension below 4.

Weller et al. use free-embedding search as an empirical upper bound on critical n: the largest corpus on which unit-norm vectors of dimension d still fit every top-2 pair under full-batch InfoNCE. The table recorded from their paper (Table 6, copied in `experiments/free-embedding/report.md`) gives critical n = 10 at d=4, 28 at d=8, and 47 at d=12. Under that curve, 46 documents are beyond what dimension 4 or 8 could fit, and they sit near dimension 12. A miss in that search is not a lower bound: the search never has to exhibit a feasible point. The moment-curve queries are feasible points in dimension 4, and on the integer nodes the margin is an integer that float64 stores exactly. The 2k+1 = 5 sign-rank style upper bound is also loose for unnormalized inner products. The fifth, constant, monomial cancels in every comparison, which is why 2k = 4 is enough.

So on LIMIT-small, dimension is the obstruction only below 4. It is not the reason a free-embedding run, or a text encoder, would fail at d=4 once this geometry is available. Adam on the sum of hinges, even in a condition-number-1 Chebyshev basis, does not find the separator in 800 steps. That is the same kind of gap as the original free-embedding curve: the optimizer stops, the configuration exists. Zero-shot encoder failure on the lexical task is a separate fact. The list-item one-hot already has Recall@2 = **1.0000** in dimension 1000, and the geometric construction shows those labels also fit in dimension 4.

