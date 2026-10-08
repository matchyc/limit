# SPLADE++ on official LIMIT (`Who likes X?`)

模型 `naver/splade-cocondenser-ensembledistil`。查询是原词，文档也是原词。limit 和 limit-small 是两次不同抽取。数字在 `experiments/splade-limit/`。small 的 SPLADE 来自同一次编码，见 `experiments/splade-gloss/`。

| 库 | 打分 | R@2 | R@10 | R@100 |
| --- | --- | ---: | ---: | ---: |
| small 46 | BM25 | 99.15 | 100 | — |
| small 46 | SPLADE 整篇 | 92.05 | 99.35 | — |
| small 46 | SPLADE 按属性句取最大 | 99.75 | 100 | — |
| small 46 | MiniLM 整篇 | 16.2 | 48.05 | — |
| small 46 | 论文 GTE-ModernColBERT | 83.5 | 97.6 | — |
| 50k | BM25 | 92.65 | 94.60 | 96.30 |
| 50k | SPLADE 整篇 | 33.8 | 58.65 | 83.55 |
| 50k | 论文 GTE-ModernColBERT | 23.1 | 34.6 | 54.8 |
| 50k | 论文 Promptriever 4096 | 3.0 | 6.8 | 18.9 |
| 50k | 论文 BM25 | 85.7 | 90.4 | 93.6 |

46 篇上，原词几乎私有，SPLADE 的 max-pool 保住一次出现，整篇 92.05，切开 99.75。低于 BM25 的那 7 个点是扩展噪声：一篇文档同时扩展 45 项喜好。

5 万篇上，同一套整篇 SPLADE 掉到 R@2 33.8。仍高于论文里的 ColBERT 和单向量，低于 BM25。较好那篇标注文档的中位名次是 3，平均名次是 142：一部分查询仍然进前几名，另一部分被同词或近邻扩展冲到很后面。
