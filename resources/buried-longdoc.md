# Buried information in long documents

Correction first: LIMIT is not a long-document benchmark. Its documents are short
profiles; it tests how many top-k subsets one vector can return. For dilution of one
span inside a long document use the work below.

## MaxP and PARADE (Dai & Callan, SIGIR 2019; Li et al., arXiv:2008.09093)

Split a doc into overlapping passages, score each independently, aggregate. MaxP (best
passage) beats FirstP (truncate) and SumP/AvgP when relevance is localized: Robust04
title nDCG@20 0.469 vs 0.444. PARADE aggregates passage CLS vectors (max, attention,
CNN, Transformer) then scores once; better when evidence spreads across passages.
ECIR 2021 reproduction: the MaxP-vs-AvgP gap is larger than originally reported.
Rule: mean over passages is strictly worse than max for localized relevance.

## Position bias and FarRelevant (Boytsov et al., 2022-2025, arXiv:2207.01262)

Benchmarks put answers early, so truncation looks fine. Forcing the relevant passage
past token 512 (FarRelevant) drops FirstP to ~random; MaxP/PARADE-attention stay
robust zero-shot. Long-context window alone does not fix buried-span retrieval.

## Pooling theory

- Pooling and Semantic Shift (arXiv:2603.21437): pooling dilutes micro semantics and
  shrinks mean pairwise distance; harm scales with semantic diversity, not length.
- Why Mean Pooling Works (arXiv:2604.27398): mean is a first-order statistic; distinct
  token sets can collide after mean. Keep second-order or multi-vector information for
  tail spans.

## DICE (Lyu et al., arXiv:2606.18781, code PunchlineAAAA/DICE)

Evidence Dilution Index = doc-vector score minus best chunk-vector score on the same
gold doc. Chunk (~1024 tokens), encode independently, aggregate chunk vectors back to
one doc vector. LongEmbed Dream backbone 63.4 -> 81.9 avg; passkey >4K 30 -> 90;
needle >4K 23 -> 74. Chunk granularity dominates; aggressive pooling reintroduces
dilution. Primary citation for "mean over full doc washes out one span".

## Late chunking (Gunther et al., Jina AI, arXiv:2409.04701)

Encode full context once (up to 8K), then mean-pool within chunk boundaries. Keeps
cross-chunk context without premature pooling. Code: jina-ai/late-chunking.

## ColBERT token pooling budget (Answer.AI, 2024)

Pooling ColBERT token vectors 2x keeps ~100% quality at half the vectors, 3x keeps
~99%, beyond that dilution returns.

## Long-doc benchmarks and what they show

- LongEmbed (Zhu et al., EMNLP 2024, arXiv:2404.12096): needle/passkey synthetics plus
  NarrativeQA/QMSum/2WikiMultihopQA. Extension to 32K helps but real tasks stay hard.
- MLDR via BGE-M3 (Chen et al., arXiv:2402.03216): one model, dense + sparse +
  multi-vector heads. nDCG@10: dense 54.9, sparse 65.0, multi-vector 60.0, combined
  65.0. Sparse alone beats dense by 10 points on long docs.
- RULER / SummHay / LoCoMo: harder needle variants (multi-needle, coverage, temporal).
  Vanilla needle is near-saturated; prefer LongEmbed/FarRelevant/MLDR for retriever
  ablations.

## Takeaway for our design

Fix menu in order: MaxP over intact chunks, PARADE-style aggregation, late chunking,
DICE chunk-aggregate, ColBERT MaxSim with pooling budget <=3x, hybrid dense+sparse+
multivector. Our first experiment below starts at step one: chunk-MaxP with the same
encoder that fails whole-doc, isolating pooling from semantics.
