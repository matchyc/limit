# Theory: exact capacity is cheap, margin is not

## LIMIT (Weller et al., ICLR 2026, arXiv:2508.21038)

Single-vector retrieval must map every query to one vector and rank by inner product.
LIMIT builds 1000 queries of the form "Who likes X?", each with exactly 2 relevant
documents out of 50k (or 46 in LIMIT-small). The 46 relevant people cover ~1000 of the
1035 possible pairs. State-of-the-art dense models score Recall@100 below 20 on the
full corpus while BM25 is near 94, and the dense qrel pattern is much harder than
random/cycle/disjoint patterns. Correlation with BEIR is near zero.

Takeaway: a simple, lexically solvable task breaks zero-shot single-vector models.
What it does not prove on its own is that dimension is the cause.

## Exact MED is Theta(k) (Wang et al., ICML 2026, arXiv:2601.20844)

For inner product, Euclidean, and cosine scoring, the minimal dimension that exactly
realizes every answer set of size up to k is Theta(k), independent of corpus size m.
Inner product needs at most 2k dimensions (cosine 2k+1). Construction: documents on a
cyclic polytope / moment curve, each query vector from a squared polynomial. For
k=2, dimension 4 exactly overfits LIMIT and LIMIT-small. Random token-sum embeddings
with no training already beat the reported 4096-d single-vector baseline at 512-d.
Code: https://github.com/zihao-wang/med

Our own check agrees: on `(t,t^2,t^3,t^4)` with query `-((x-a)(x-b))^2`, LIMIT-small is
1000/1000 strict with integer margin >= 1; unit-normalized margins shrink to ~1e-6.

## Robust MED brings m back (same paper, Section 4)

Require unit vectors plus a normalized score gap epsilon. Feasibility is capped by
epsilon*(m,k) ~ 1/sqrt(k); above it no dimension works. At the feasible scale
c/sqrt(k), a Gaussian centroid construction gives O(k^2 log m) dimensions. The same
4-d witness has tiny margins and poor conditioning: it rules out "no exact geometry"
but does not give a deployable index.

## Revised LIMIT (arXiv:2508.21038v2)

Main theorem is now a sphere-packing lower bound with margin gamma:

d >= log C(n,k) / log(1 + 1/gamma)

At gamma=0.1, k=2, n=1e6 the bound is ~12, not thousands. The paper still reports that
InfoNCE free embeddings need ~4.5x the bound and extrapolates a cubic critical-n
curve. Appendix D concedes the no-margin sign-rank bound depends only on k (Alon,
Frankl, Rodl 1985, upper bound 2k) and that the construction needs infinite
precision in general.

## The gap both papers leave open

Neither paper measures, on one fixed qrel graph, the minimal dimension that holds a
fixed normalized margin with a margin-seeking optimizer. That table (epsilon_star
feasibility, packing lower bound, achieved dimension with hinge loss) is the missing
measurement behind our direction 1 in `../notes/research-directions.md`.
