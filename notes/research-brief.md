# LIMIT 研究准备

单向量 embedding 检索能不能表示任意 top-k 相关文档组合，以及 LIMIT 上的失败到底来自维度不够，还是来自学习、泛化和打分对齐。

## 本地材料

代码来自 [matchyc/limit](https://github.com/matchyc/limit)，上游是 [google-deepmind/limit](https://github.com/google-deepmind/limit)（`upstream` remote）。论文在 `papers/`：

| 文件 | 文献 |
| --- | --- |
| `papers/weller2025-limit-arxiv.pdf` | Weller et al., arXiv:2508.21038v2。ICLR 2026。 |
| `papers/weller2026-limit-iclr.pdf` | 同一篇的 ICLR 会议版。 |
| `papers/wang2026-2k-med-arxiv.pdf` | Wang et al., arXiv:2601.20844。反驳“几何容量不够”。 |
| `papers/mihir2026-single-vector-arxiv.pdf` | Archish et al., arXiv:2603.29519。k-sparse 时 2k+1 维够用，并把失败归因于 domain shift 与 drowning。 |

Hugging Face 数据：[orionweller/LIMIT](https://huggingface.co/datasets/orionweller/LIMIT)、[orionweller/LIMIT-small](https://huggingface.co/datasets/orionweller/LIMIT-small)。

## 任务本身

每条 query 是 `Who likes <attribute>?`。每篇文档是一个人名加上一长串 “likes A, B, and C.”。相关文档必须在名单里包含该属性。这是集合包含，不是语义改写。

| 切分 | queries | 每条 query 的相关文档 | corpus |
| --- | --- | --- | --- |
| `data/limit` | 1000 | 2 | 50000 |
| `data/limit-small` | 1000 | 2 | 46（只留被某条 query 命中的人） |

qrels 不是随机抽样。作者取 `k=2`，用最小的 `n` 使 `C(n, 2) >= 1000`，即 `n=46`（`C(46,2)=1035`），再取其中 1000 个 pair。这 46 个人两两组合几乎都被问到。其余 49954 篇是无关填充，每人属性数补到与相关文档一样长（约 50 个属性）。属性表来自 Gemini 2.5 Pro，清洗到约 1850 个，并避开 BM25 会混淆的重复和上下位词。

生成逻辑在 `code/generate_limit_dataset.ipynb`。自由向量实验在 `code/free_embedding_experiment.py`：对 query/doc 向量本身做全 batch InfoNCE，找给定维度 `d`、`k=2` 时还能 100% 分开的临界文档数 critical-n。

## 原文的论点

检索目标写成二值相关矩阵 `A`。单向量内积检索要求存在低秩分数矩阵，使每一行里相关文档的分数高于不相关文档。这等价于 row-wise order-preserving rank，并被 sign-rank 夹住：

`signrank(2A - 1) - 1 <= rank_rt(A) <= signrank(2A - 1)`

因此，固定的 query–document 相关矩阵若 sign-rank 高，任何单向量模型都无法用小的 `d` 精确实现全部 top-k 集合。作者用 free embedding 把这个界做成可优化的上界：若梯度下降都分不开，真实语言模型更分不开。

Free embedding（unit norm，Adam，InfoNCE，直接在目标 qrels 上过拟合）给出的临界 `n` 大致是 `d` 的三次多项式。外推：`d=512` 约 50 万文档，`d=4096` 约 2.5 亿。实验网格只到几十维，外推不是证明。

LIMIT 上的实测（原文 Table 4 / Table 5，满分 100）：

| 模型 | 设定 | Recall@2 | Recall@10 | Recall@100 |
| --- | --- | --- | --- | --- |
| Promptriever Llama3 8B, 4096 | full 50k | 3.0 | 6.8 | 18.9 |
| GritLM 7B, 4096 | full | 2.4 | 4.1 | 12.9 |
| E5-Mistral 7B, 4096 | full | 1.3 | 2.2 | 8.3 |
| GTE-ModernColBERT | full | 23.1 | 34.6 | 54.8 |
| BM25 | full | 85.7 | 90.4 | 93.6 |
| Promptriever, 4096 | small, 46 docs | 54.3 | 90.0 | — (R@20 = 97.7) |
| GTE-ModernColBERT | small | 83.5 | 97.6 | — (R@20 = 99.1) |
| BM25 | small | 97.8 | 100 | — |

同为 50k / 1k queries / k=2，把 qrels 从 dense pair 换成 random、cycle 或 disjoint 后，E5-Mistral 4096 的 Recall@100 从 4.8 升到约 40。难的是组合密度，不是句子表面。

作者还做了 domain-shift 对照：在同分布训练属性上微调 ModernBERT-embed，LIMIT 测试 Recall@10 最高约 2.8；直接在测试属性上过拟合则 Recall@2 到 85–96。Gemini 2.5 Pro 把 46 篇文档一次放进上下文，1000 条 query 全部答对。结论是：cross-encoder 不受这个向量维数约束；sparse 模型靠极高维词表躲开它；multi-vector 更好，但仍未解 full LIMIT。

## 后续工作改写了“维数不够”这个解释

两条独立的后续论文都指出：LIMIT 的相关矩阵每行只有 `k=2` 个 1。对这种 k-sparse 相关，精确 top-k 不需要随文档数增长的维度。

Wang et al. (2601.20844) 定义 Minimal Embeddable Dimension。对内积、欧氏距离、余弦，精确实现“每个大小至多 k 的子集都能被某条 query 检索出来”的维度是 `Theta(k)`，与宇宙大小 `m` 无关。内积下界 `k-1`，上界 `2k`；余弦上界 `2k+1`。`k=2` 时 cyclic polytope 在 4 维就能精确过拟合 LIMIT 和 LIMIT-small。他们用无训练的随机加性构造，512 维已超过原文最好的单向量 LLM；4096 维 random additive 在 LIMIT-small / LIMIT 上 top-2 recall 约 0.95 / 0.70。带间隔的 Robust MED 才重新让 `m` 以 `O(k^2 log m)` 出现，可行间隔上限约 `1/sqrt(k)`。

Archish et al. (2603.29519) 用 Alon et al. 2016：k-sparse 相关矩阵的 sign-rank 有上界，`2k+1=5` 维就够表示 LIMIT 这张固定矩阵。他们把零样本失败改写成：长属性列表是 domain shift，余弦和集合包含不对齐。微调后单向量 Recall@10 从约 1% 到约 40%，multi-vector 从约 40% 到约 98%；单向量在 MS MARCO 上遗忘超过 40 个 Recall@100 点，multi-vector 约 1 个点。drowning：无关文档变多时，单向量负例压过正例的概率约 `exp(-Theta(d/n))`，multi-vector 约 `exp(-Theta(d/log n))`。控制 tokenizer 和近义属性只能把单向量 Recall@10 从约 1% 抬到约 3%，multi-vector 的 Recall@2 可以从 27% 到 96%。

因此原文 free-embedding 的三次曲线更像优化失败（InfoNCE、单位范数、早停、数值间隔），不是几何不可表示。原文自己也写了 sign-rank 很难算，dense qrels “更难”是直觉而不是定理。

仍然成立、而且对向量检索更有用的部分：

- 零样本单向量模型在这个词面完全可解的任务上接近失败，BM25 接近完美。
- 组合越密越难，和 BEIR 分数几乎无关。
- 真实编码器比自由向量更受语言参数化约束。测试集过拟合能学会这些 token，换一组属性就几乎不会。
- 指令检索若没有词面重叠，BM25 这条退路也不存在。原文把 hybrid / 推理型 sparse 留作开放问题。

## 建议先做的实验

1. 在 LIMIT-small 上复现一个小模型（例如 `all-MiniLM-L6-v2`）的 Recall@2/10，确认 MTEB `LIMITSmallRetrieval` 能跑通。评测钉在 README 里的 MTEB `w_limit` 分支。
2. 用 cyclic polytope 或质心 query 在 `d=4` 上精确拟合 LIMIT-small 的 qrels，量一下正负分数间隔。这直接检验“容量 vs 间隔”。
3. 把 `free_embedding_experiment.py` 的 InfoNCE 和已知可解构造对照，看临界 `n` 是优化器停了，还是约束真的不可行。
4. 分开测三件事：属性是否在词面上出现、文档列表长度、无关文档数量。这对应 lexical shortcut、domain shift 和 drowning。
5. 只有在上面把“表示”和“泛化”拆开之后，再谈索引、量化或多向量 late interaction。ANN 不能修复一个根本没有把相关文档排到前面的向量。

`code/requirements.txt` 依赖 JAX CUDA 12 和 `mteb @ w_limit`。自由向量实验示例：

```bash
python code/free_embedding_experiment.py --d=4 --k=2 --enable_critical_n_search=11 \
  --results_output_path='d=4_k=2.json' --device=gpu
```

## 引用

```bibtex
@inproceedings{weller2026theoretical,
  title={On the Theoretical Limitations of Embedding-Based Retrieval},
  author={Weller, Orion and Boratko, Michael and Naim, Iftekhar and Lee, Jinhyuk},
  booktitle={International Conference on Learning Representations},
  year={2026},
  url={https://arxiv.org/abs/2508.21038}
}
```
