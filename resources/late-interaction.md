# Late interaction / multi-vector retrieval

Core idea: keep one vector per token (or chunk) and defer matching to query time.
Query token i scores max over document tokens j (MaxSim / Chamfer), then sums. One
strong local match can rescue a document that mean pooling would bury. Price: many
vectors per document unless compressed.

## ColBERT (Khattab & Zaharia, SIGIR 2020, arXiv:2004.12832)

Query and doc encoded separately by BERT into token vectors; score is MaxSim.
MS MARCO MRR@10 36.0, R@50 82.9. ~170x faster than a cross-encoder reranker but the
token-ANN + gather stage is slow at scale and limited to ~512 tokens per doc.
Code: https://github.com/stanford-futuredata/ColBERT

## ColBERTv2 + PLAID (Santhanam et al., NAACL 2022 / CIKM 2022, arXiv:2112.01488 / 2205.09707)

Same scoring; residual compression (centroid id + 1-2 bit residual) shrinks the MS
MARCO index from ~154 GiB to ~16-25 GiB. Denoised distillation helps zero-shot
(BEIR avg ~50 nDCG@10). PLAID serves it in ~38ms GPU via centroid interaction and
pruning. Recall-critical workloads should raise nprobe and lower pruning thresholds,
because pruning eats rare-constraint tokens first.

## GTE-ModernColBERT (LightOn, 2025, open release, no paper)

ColBERT architecture on ModernBERT with up to 8k-token docs, distilled from MS MARCO
with a reranker teacher. BEIR avg ~54.7. Most directly relevant to buried attributes:
the whole long list stays in one encoding, but 8k context multiplies vectors per doc,
so pooling 2-3x or PLAID-style filtering is expected.

## XTR (Lee et al., NeurIPS 2023, arXiv:2304.01982)

Trains tokens to be retrieved in stage 1, then scores only retrieved tokens with
imputation for the rest. BEIR-13 up to 52.7. Best paraphrase story here, but if the
constraint token is missed at stage 1 there is nothing to rescore.

## MUVERA (Jayaram et al., NeurIPS 2024, arXiv:2405.19504)

Hashes the token set into a fixed-dimension encoding whose inner product approximates
Chamfer similarity, so multi-vector quality runs on single-vector ANN infra, followed
by exact rescore on a shortlist. Needs high encoding dim (>=5-10k) plus rescore for
constraint-critical queries; the sketch alone dilutes one token among fifty.
Code: https://github.com/google/muvera

## Takeaway for our design

MaxSim is the canonical "no early pooling" fix and GTE-ModernColBERT removes the
512-token chunking excuse, but per-token storage and pruning risk remain. Our unified
score should keep the max-over-local-evidence behavior while staying near
single-vector cost. Operating rules: intact chunks, recall-tuned pruning, exact
rescore on top-N.
