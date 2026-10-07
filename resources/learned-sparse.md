# Learned sparse retrieval

All methods below keep exact lexical grounding on an inverted index, so a rare buried
phrase is never fully averaged away. Ranking for paraphrase tolerance tracks whether
query-side learned expansion exists.

## SPLADE / SPLADEv2 (Formal et al., SIGIR 2021 / arXiv:2109.10086)

BERT MLM head maps every token to a vocabulary-size logit vector; ReLU +
log-saturation aggregates over tokens with FLOPS/L1 sparsity control. Both query and
document can gain terms absent from the text. MS MARCO dev MRR@10 0.322 (v1), 0.340
(max-pool), 0.368 distilled, recall@1000 up to ~0.98. BEIR avg ~0.50, above BM25 and
above weighting-only sparse. Max-pool keeps one buried mention from being washed out.
Code: https://github.com/naver/splade

## BT-SPLADE / efficient SPLADE (Lassance et al., SIGIR 2022, arXiv:2207.03834)

Decoupled doc/query encoders with a tiny query encoder. Keeps ~97-103% of SPLADE
quality at BM25-adjacent latency (single-digit ms). The deployable sparse choice for
first-stage candidate generation.

## DeepCT (Dai & Callan, SIGIR 2020, arXiv:1910.10687) and HDCT for long docs

BERT predicts a TF-like integer weight per occurring token; BM25 runs unchanged.
MS MARCO MRR@10 ~0.244 vs BM25 ~0.184. Good for buried exact phrases, useless for
paraphrase (no expansion). Cheapest upgrade to an existing BM25 stack.

## DeepImpact (Mallia et al., SIGIR 2021, arXiv:2104.12016)

DocT5Query expansion plus learned per-token impacts, 8-bit quantized. MS MARCO
MRR@10 0.326, recall@1000 0.948. Paraphrase coverage only through what T5 injected.

## uniCOIL / COIL (Lin & Ma; Gao et al., 2021, arXiv:2106.14807 / 2104.07186)

Decomposes sparse-learned into expansion + weighting. uniCOIL-noexp 0.315, +T5 0.352
on MS MARCO. Cleanest testbed: weighting alone vs weighting plus expansion.

## SPARTA (Zhao et al., NAACL 2021, arXiv:2009.13013)

Asymmetric: query stays bag-of-words, passage encoder predicts weights over query-term
space. Strong in-domain QA, ~20% below BM25 zero-shot on BEIR. Brittle under
vocabulary shift.

## Takeaway for our design

Use DistilSPLADE-max / SPLADE++ / BT-SPLADE-L as the sparse channel when paraphrased
constraints matter (target recall@1000 >= 0.97). Weighting-only methods are not
enough for wording changes. Sparse keeps the verbatim constraint; it does not supply
the semantic half of the unified score.
