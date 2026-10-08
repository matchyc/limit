# SPLADE 接不住无共同实词的释义查询

模型是公开的 SPLADE++：`naver/splade-cocondenser-ensembledistil`。库是 LIMIT-small，46 篇，1000 道原词查询，其中 126 道换成手写释义，释义和属性没有共同实词。整篇最长 173 个 token，没有截断。数字在 `experiments/splade-gloss/`。

第二件问题是：查询写定义，文档里仍是原词。BM25 的 Recall@2 是 3.57。MiniLM 块最大是 24.21。

## 释义查询

| 打分 | R@2 | R@10 | 两篇都在前 2 |
| --- | ---: | ---: | ---: |
| BM25 | 3.57 | 24.21 | 0.0 |
| SPLADE 整篇 | 8.33 | 44.44 | 0.79 |
| SPLADE 按属性句取最大 | 16.67 | 53.97 | 9.52 |
| MiniLM 按属性句取最大（先前） | 24.21 | 62.70 | 11.11 |

查询扩展里出现属性的某个实词：38/126（30.16%）。出现时整篇 R@2 是 14.47；不出现时是 5.68。Disco Music 算命中，是因为扩展出了 music，不是 disco。Ham、Joshua Trees、Havarti、Chairs、Limes、Barley 都没有把原词扩展出来。

Ham 的查询 “Who likes cured pork cut from a thigh?” 扩展的是 thigh、pork、cured，没有 ham。两篇标注文档排在第 25、第 37。Joshua Trees 扩展的是 fork、crown、yu/##cca，没有 joshua。Havarti 扩展的是 cheese、holes、mild，没有 havarti。

反过来：这 126 个属性，查询仍写原词，文档改成释义。BM25 从 96.83 掉到 4.37。SPLADE 从 85.32 掉到 12.3。文档侧扩展同样没有把定义写回原词。

## 原词对照

| 打分 | R@2 |
| --- | ---: |
| BM25 | 99.15 |
| SPLADE 整篇 | 92.05 |
| SPLADE 按属性句取最大 | 99.75 |

原词还在时，SPLADE 的 max-pool 能保住一次出现，切开之后和 BM25 同一档。整篇 92.05 低于 BM25，是 45 项喜好同时扩展带来的噪声，不是第二件问题。

## 结论

SPLADE 解决不了第二件问题。它多出来的是 BERT 词表上的 MLM 扩展，扩展的是定义里已经有的词（pork、cheese、citrus），不是定义所指的那个专名（Ham、Havarti、Limes）。扩展命中原词的那 30% 也只有 14.47，仍低于 MiniLM 块最大的 24.21。
