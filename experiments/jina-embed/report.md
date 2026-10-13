# Jina Embeddings v5 text-small (retrieval) on official LIMIT

Model `jinaai/jina-embeddings-v5-text-small-retrieval` (1024-d) on cuda. Env `experiments/zero-shot-embed/.venv`. 114.7s.

Jina v5-text-small (Feb 2026) retrieval-merged checkpoint, also the Text Embeddings Inference (TEI) weights. Queries use prompt Query:, documents Document:.

## limit-small (46 documents)

| score | R@2 | R@10 | R@20 | both@2 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 99.15 | 100.0 | 100.0 | 98.4 |
| Jina v5-text-small retrieval | 16.55 | 46.8 | 68.45 | 4.0 |
| Qwen3-Emb 0.6B (prior) | 14.45 | — | — | — |
| MiniLM (prior) | 16.2 | — | — | — |
| paper Promptriever 4096 | 54.3 | — | — | — |
| SPLADE++ whole (prior) | 92.05 | — | — | — |

## limit (50k documents)

| score | R@2 | R@10 | R@20 | R@100 | both@2 | better gold median/mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 92.65 | 94.6 | 95.15 | 96.3 | 91.0 | 1.0 / 11.84 |
| Jina v5-text-small retrieval | 0.8 | 1.55 | 1.7 | 3.75 | 0.1 | 4981.5 / 8216.31 |
| Qwen3-Emb 0.6B (prior) | 0.95 | 1.8 | 2.2 | 3.85 | 0.2 | 6999.0 / 10432.02 |
| SPLADE-v3 DistilBERT (prior) | 38.9 | — | — | — | — | — |
| paper Promptriever 4096 | 3.0 | — | — | — | — | — |
| paper E5-Mistral 7B | 1.3 | — | — | — | — | — |
| paper GTE-ModernColBERT | 23.1 | — | — | — | — | — |

## Cases (50k)

Top-8 is almost entirely name-surface matches. Golds sit thousands of ranks down; none of the top hits contain the queried like.

- `Who likes Art History?` golds [14682, 26691]; top1 `Kristofer Artale`; also Arta, Artelia, Art Liem, Bart Huisingh. Cosine 0.53 vs golds ~14k/26k.
- `Who likes Honey?` golds [720, 8507]; top1 `Berlin Honeyfield`; also Honig, Hornseth (Honey Bees), Honigford, Honegger.
- `Who likes Psychology?` golds [5373, 17411]; top1 `Emilio Psomas`; also Pinsky, Psaras, Pasion. No likes overlap.
- `Who likes Go?` golds [22037, 2448]; top1 `Goebel Goetten`; then Goerl, Goelz, Goetz, Goza.
- `Who likes Birch Trees?` golds [31669, 25223]; top1 `Kenneth Birchman`; then Birkenholz, Birtha, Birkhoff, Birck.

## Takeaway

Jina v5-text-small is the current small Jina retrieval model (677M, distilled from Qwen3-Embedding-4B). The TEI serving weights are this merged retrieval checkpoint. On LIMIT it does not beat MiniLM on the 46-doc set, and on 50k it collapses like Qwen3-0.6B / E5-Mistral: dense cosine latches onto person-name substrings (`Artale`, `Honeyfield`, `Psomas`) instead of a buried exact like. Single-vector LLM embeddings do not fix LIMIT.

