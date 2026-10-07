# Resource library: buried detail vs semantics in first-stage retrieval

Question: dense single-vector retrieval washes out a local constraint buried in a long
document. BM25 keeps it when words match and misses it when wording changes. Late
fusion (e.g. RRF) cannot recover a constraint missed by both channels. What is missing
is one first-stage score that preserves a local span, matches it under paraphrase, and
still ranks ordinary semantic similarity, cheap enough for a shortlist of ~100.

## Map

- `theory-limits.md` — LIMIT, exact MED (2k), robust MED, why exact capacity is cheap
  and margin is not.
- `late-interaction.md` — ColBERT family, PLAID, XTR, MUVERA, GTE-ModernColBERT.
- `learned-sparse.md` — SPLADE family, DeepCT/DeepImpact, uniCOIL/COIL, SPARTA.
- `hybrid-fusion.md` — RRF, convex combination, learned/query-gated fusion, hybrid
  indexes, joint training (CLEAR, RoC, SPLATE/CITADEL), and the union-recall ceiling.
- `buried-longdoc.md` — MaxP/PARADE, pooling theory, late chunking, DICE, long-doc
  benchmarks, and what to cite for mean-pooling dilution.
- `papers.json` — machine-readable index (title, authors, year, venue, arxiv, code,
  tags) for the core entries.

## How to use

Each file ends with a takeaway for our design in `../notes/unified-score-design.md`.
Start with `theory-limits.md` for what is already settled, then `hybrid-fusion.md`
section F for why two-list fusion is not the answer, then `buried-longdoc.md` for the
fix menu. Experiment code lives in `../experiments/`.
