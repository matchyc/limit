# Hybrid fusion and why two lists are not enough

## RRF (Cormack, Clarke, Buttcher, SIGIR 2009)

score(d) = sum over systems of 1/(k + rank), k=60 default. Rank-only, training-free,
no calibration. In production via Pyserini, OpenSearch, Elasticsearch. Beats the best
single system by a few MAP points on TREC, but discards score margins and cannot
promote a document retrieved by neither channel.

Query-adaptive RRF (2026) reweights per query (rare-token query trusts sparse,
paraphrastic query trusts dense) yet stays rank-only and bounded by union recall.

## Convex combination (Wang et al. ICTIR 2021; Chen et al. TOIS, arXiv:2210.11934)

f = alpha * dense + (1-alpha) * sparse, after min-max or z normalization (BM25 is
unbounded, dense cosine is bounded). Elastic replication: oracle alpha gives +6% over
sparse-only and +24% over BM25-only; RRF gives +1.4% / +18%. Tuned convex beats RRF
in- and out-of-domain but needs labels and stable normalization. Pinecone single-index
hybrid is the production form of the same formula.

## Learned and query-gated fusion

LTR cascades (LambdaMART over lexical + dense features) gain up to ~+11% nDCG@10 at
small latency cost. Per-query alpha(q) from the query embedding beats static alpha by
a few points with enough graded pairs. Both still rerank the union.

## Hybrid indexes

OpenSearch hybrid search (normalize then combine, min-max + mean best in their tests,
+8-12% vs text search). Vespa unifies sparse wand/weakAnd with dense nearestNeighbor
plus phased ranking in one request. These fix plumbing, not the ceiling below.

## Distillation (Hofstatter et al., Margin-MSE; TAS-B, SIGIR 2021)

Matches teacher-student score margins to inject cross-encoder quality into a
bi-encoder. Produces the strong dense side of many hybrids. Still one channel of a
two-list system.

## The union-recall ceiling (the key section)

Late fusion operates on U^k(q) = TopK_sparse union TopK_dense. Fused recall <= union
recall. If gold d* is outside the union, no weight, rank formula, or reranker sees it.
Two cutoffs mean two chances to lose. Sparse and dense fail differently so unions
usually help, but a constraint needing both exactness and semantics can be missed by
both at once; fusion then reorders distractors. Practitioner rule: measure union
recall at candidate depth before touching fusion weights. ORE (Rathee et al., SIGIR
2025) is the non-joint alternative: bandit-style exploration over a large pool instead
of fusing two fixed top-ks (up to +58% recall on TREC DL22 at fixed ranker budget).

## Single-score joint training (the escape)

- CLEAR (Gao et al., ECIR 2021, arXiv:2004.13969): train dense to fix BM25 mistakes
  with a residual margin. First joint template.
- RoC (Lee et al., ACL 2023): residual complementarity is incomplete; two-level
  orthogonality decorrelates channels so dense adds new coverage. Use RoC for "do the
  channels actually complement" rather than raw hybrid NDCG.
- Luan et al. (TACL 2021): theory for fixed-length dual-encoder capacity vs margin vs
  doc length; hybridized multi-vector at/near top everywhere.
- CITADEL (ACL 2023) / SPLATE (arXiv:2404.13950) / ColBERT-serve: one transformer pass
  yields sparse candidates plus MaxSim rescore (SPLATE adapter ~0.6M params, 50 reranks
  at PLAID quality). Closest to a single-score joint system: one encoding, no two-list
  union to miss.

## Takeaway for our design

RRF is the safe default without labels; tuned convex is stronger with ~300 labeled
queries; learned gating is best with logs. None of them removes the ceiling. Our
unified score must be joint at encoding or scoring time (CLEAR/RoC/SPLATE direction),
or pair a joint scorer with ORE-style pool exploration.
