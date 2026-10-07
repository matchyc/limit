# Unified first-stage score: design

Goal: one first-stage score that (1) preserves a local span buried in a long
document, (2) matches it under paraphrase, (3) still ranks ordinary semantic
similarity, (4) returns a shortlist of ~100 cheaply. Late fusion of two independent
top-k lists (RRF, convex) is out of scope as the answer: it cannot recover a
constraint missed by both channels (see `../resources/hybrid-fusion.md` section on
the union-recall ceiling).

## Approach A: chunk-MaxP with the same encoder (recommended first)

Score each document as the max over its chunks, not the mean over the whole text:

score(q, d) = max_c cos(enc(q), enc(chunk_c))

Chunks are attribute spans ("<Name> likes <attr>.") on LIMIT, passages on real docs.
One encoder, one ANN index over chunks, doc score by max. Cost is ~chunks-per-doc
more vectors; no second model, no second list to fuse. This isolates pooling from
semantics: same weights, only the aggregation changes. MaxP/PARADE and DICE say max
beats mean for localized relevance; late chunking says keep full-doc context when
encoding chunks.

Limits: chunking can split a constraint; max-over-chunks still uses one cosine per
chunk, so a paraphrased span needs the encoder to match it. That is exactly what the
experiment below measures.

## Approach B: single-encoder dual head (research goal)

One transformer pass yields a dense vector plus a sparse lexical vector
(SPLATE/CITADEL direction), scored jointly: S = f(dense) + g(sparse). No two-list
union to miss. Dense head carries paraphrase, sparse head carries the verbatim span,
trained with a complementarity objective (CLEAR residual, RoC orthogonality) so the
heads add coverage instead of re-covering the same hits. Heavier to train; needs a
paraphrase split of LIMIT (direction 3) to prove the joint score beats either head.

## Approach C: pooled token MaxSim-lite

Keep per-token vectors but pool 2-3x at index time (Answer.AI budget: 2x keeps full
quality, 3x ~99%), serve with PLAID-style centroid pruning tuned for recall, exact
MaxSim rescore on top-N. Closest to ColBERT quality at near-single-vector
footprint. More infra than A (token index, pruning tuning) for gains that matter most
when spans are paraphrased at token level.

## Decision

Run A first: same MiniLM that fails whole-doc, chunked MaxP on LIMIT-small. If max
recovers most of the gap to the word-overlap ranker, pooling is the isolated cause
and B/C buy paraphrase on top. If max still fails, the encoder itself cannot match
the span and B must come first. Next steps after A: paraphrase split (no shared rare
token), then fixed-margin table (direction 2), then a dual-head prototype (B).
