# Qwen3-Embedding-0.6B on official LIMIT queries

Model `Qwen/Qwen3-Embedding-0.6B` (1024-d) on cuda. Interpreter is `experiments/zero-shot-embed/.venv` (sentence-transformers 6.1, transformers 5.18, torch 2.10+cu128). 107.7s.

Queries use either the model's default retrieval prompt or a LIMIT-specific instruction. Documents are encoded with no prompt. Cosine similarity, official `Who likes X?` queries.

## limit-small (46 documents)

| score | R@2 | R@10 | R@20 | both@2 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 99.15 | 100.0 | 100.0 | 98.4 |
| Qwen3-Emb 0.6B default prompt | 14.45 | 41.6 | 63.05 | 3.2 |
| Qwen3-Emb 0.6B LIMIT instruction | 13.6 | 39.35 | 62.85 | 3.2 |
| MiniLM (prior) | 16.2 | — | — | — |
| paper Promptriever 4096 | 54.3 | — | — | — |
| SPLADE++ whole (prior) | 92.05 | — | — | — |

## limit (50k documents)

| score | R@2 | R@10 | R@20 | R@100 | both@2 | better gold median/mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 92.65 | 94.6 | 95.15 | 96.3 | 91.0 | 1.0 / 11.84 |
| Qwen3-Emb 0.6B default | 0.95 | 1.8 | 2.2 | 3.85 | 0.2 | 6999.0 / 10432.02 |
| Qwen3-Emb 0.6B LIMIT inst. | 0.9 | 1.45 | 2.0 | 3.5 | 0.2 | 7994.0 / 10847.71 |
| SPLADE++ (prior) | 33.8 | — | — | — | — | — |
| SPLADE-v3 DistilBERT (prior) | 38.9 | — | — | — | — | — |
| paper Promptriever 4096 | 3.0 | — | — | — | — | — |
| paper E5-Mistral 7B | 1.3 | — | — | — | — | — |
| paper GTE-ModernColBERT | 23.1 | — | — | — | — | — |

## Cases (50k, default prompt)

- `Who likes Art History?` golds [37656, 38980]; top1 `Dorthy Skaria` gold=False hits=[]
- `Who likes Honey?` golds [2983, 21507]; top1 `Sherry Hornseth` gold=False hits=['Honey Bees']
- `Who likes Psychology?` golds [9208, 1593]; top1 `Stetson Pollina` gold=False hits=[]
- `Who likes Go?` golds [24615, 6628]; top1 `Goebel Goetten` gold=False hits=[]
- `Who likes Birch Trees?` golds [46387, 41693]; top1 `Alpha Birr` gold=False hits=[]

