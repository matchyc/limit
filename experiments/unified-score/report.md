# Chunk-MaxP vs whole-doc (LIMIT-small, MiniLM-L6-v2)

chunks: 2072 over 46 docs

| scoring | R@2 | R@10 | R@20 | both-in-top2 |
| --- | ---: | ---: | ---: | ---: |
| whole-doc | 16.2 | 48.05 | 69.45 | 2.3 |
| chunk-max | 98.1 | 100.0 | 100.0 | 96.3 |

device=cuda, seconds=2.7
