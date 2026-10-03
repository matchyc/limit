#!/usr/bin/env python3
"""Zero-shot cosine retrieval with all-MiniLM-L6-v2 on LIMIT.

Works only on limit-small qrels. The drowning curve keeps those 46 documents
and adds a nested, seed-0 subset of data/limit/corpus.jsonl.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path("/mnt/raid0nvme0/yangshen/meng/vector_search/limit")
OUT = ROOT / "experiments" / "zero-shot-embed"
SMALL = ROOT / "data" / "limit-small"
FULL = ROOT / "data" / "limit"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
ADDED_SIZES = (0, 100, 1000, 10000, 50000)
DIAG_N = 200
SEED = 0
WORD_RE = re.compile(r"[A-Za-z0-9]+")
KS_SMALL = (2, 10, 20)
KS_CURVE = (2, 10)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def doc_text(row: dict) -> str:
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    if title and text:
        return f"{title} {text}"
    return title or text


def attribute_of(query_text: str) -> str:
    prefix = "Who likes "
    if not (query_text.startswith(prefix) and query_text.endswith("?")):
        raise ValueError(f"unexpected query text: {query_text!r}")
    return query_text[len(prefix) : -1]


def content_words(text: str) -> set[str]:
    return {w.lower() for w in WORD_RE.findall(text) if len(w) > 3}


def id_key(doc_ids: list[str]) -> np.ndarray:
    order = np.argsort(np.asarray(doc_ids), kind="mergesort")
    key = np.empty(len(doc_ids), dtype=np.int32)
    key[order] = np.arange(len(doc_ids), dtype=np.int32)
    return key


def gold_ranks(scores: np.ndarray, gold_index: np.ndarray, key: np.ndarray) -> np.ndarray:
    """1-indexed ranks of gold docs. Higher score is better; ties break by doc id.

    scores: (n_queries, n_docs)
    gold_index: (n_queries, n_gold)
    key: alphabetical rank of each doc, smaller is better on a tie
    """
    n_queries, n_gold = gold_index.shape
    ranks = np.empty((n_queries, n_gold), dtype=np.int32)
    g_scores = np.take_along_axis(scores, gold_index, axis=1)
    g_keys = key[gold_index]
    chunk = 20
    for start in range(0, n_queries, chunk):
        end = min(start + chunk, n_queries)
        sl = scores[start:end]
        gs = g_scores[start:end]
        gk = g_keys[start:end]
        higher = (sl[:, None, :] > gs[:, :, None]).sum(axis=-1)
        earlier_tie = (
            (sl[:, None, :] == gs[:, :, None]) & (key[None, None, :] < gk[:, :, None])
        ).sum(axis=-1)
        ranks[start:end] = higher + earlier_tie + 1
    return ranks


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    """Macro-average of |gold in top k| / |gold|. Every query has the same |gold|."""
    return float((ranks <= k).mean())


def top_wrong_index(scores: np.ndarray, gold_index: np.ndarray, key: np.ndarray) -> np.ndarray:
    masked = scores.copy()
    for i in range(scores.shape[0]):
        masked[i, gold_index[i]] = -np.inf
    max_s = masked.max(axis=1)
    is_max = masked == max_s[:, None]
    penalty = np.where(is_max, key[None, :], np.iinfo(np.int32).max)
    return penalty.argmin(axis=1).astype(np.int32)


def pack_recall(fraction: float) -> dict:
    return {
        "fraction": round(float(fraction), 6),
        "percent": round(float(fraction) * 100, 4),
    }


def self_check() -> None:
    scores = np.array(
        [
            [0.9, 0.9, 0.1],
            [0.1, 0.2, 0.9],
        ],
        dtype=np.float32,
    )
    ids = ["A", "B", "C"]
    key = id_key(ids)
    gold = np.array([[0, 1], [2, 0]])
    ranks = gold_ranks(scores, gold, key)
    # Query 0: A ties B at 0.9, A id wins, so A rank 1, B rank 2, C rank 3.
    assert ranks[0].tolist() == [1, 2], ranks[0]
    # Query 1: C=0.9 rank 1, B=0.2 rank 2, A=0.1 rank 3.
    assert ranks[1].tolist() == [1, 3], ranks[1]
    assert abs(recall_at_k(ranks, 2) - 0.75) < 1e-9
    wrong = top_wrong_index(scores, gold, key)
    assert wrong.tolist() == [2, 1], wrong


def embed(model: SentenceTransformer, texts: list[str], batch_size: int = 512) -> np.ndarray:
    arr = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device="cuda",
    )
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise RuntimeError(f"expected 2d embeddings, got {arr.shape}")
    return arr


def truncation_stats(model: SentenceTransformer, texts: list[str]) -> dict:
    tokenizer = model.tokenizer
    lengths = []
    # Count tokens the same way the model truncates: no truncation, special tokens on.
    for start in range(0, len(texts), 1024):
        chunk = texts[start : start + 1024]
        enc = tokenizer(
            chunk,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )
        lengths.extend(len(ids) for ids in enc["input_ids"])
    lengths_arr = np.asarray(lengths, dtype=np.int32)
    max_len = int(model.max_seq_length)
    return {
        "max_seq_length": max_len,
        "n": int(lengths_arr.size),
        "min": int(lengths_arr.min()),
        "median": int(np.median(lengths_arr)),
        "max": int(lengths_arr.max()),
        "n_truncated": int((lengths_arr > max_len).sum()),
        "frac_truncated": round(float((lengths_arr > max_len).mean()), 6),
    }


def cosine(query_emb: np.ndarray, doc_emb: np.ndarray) -> np.ndarray:
    # Both rows are L2-normalized, so the product is cosine similarity.
    q = torch.from_numpy(query_emb).to(device="cuda", dtype=torch.float32)
    d = torch.from_numpy(doc_emb).to(device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        sims = q @ d.T
    out = sims.detach().cpu().numpy()
    del q, d, sims
    torch.cuda.empty_cache()
    return out


def main() -> None:
    self_check()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.perf_counter()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; refusing to run on CPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected a single visible GPU after CUDA_VISIBLE_DEVICES=1, got {torch.cuda.device_count()}"
        )
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)

    t_load = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    model_load_s = time.perf_counter() - t_load
    dim = int(model.get_embedding_dimension())

    queries = load_jsonl(SMALL / "queries.jsonl")
    small_docs = load_jsonl(SMALL / "corpus.jsonl")
    full_docs = load_jsonl(FULL / "corpus.jsonl")
    qrels = load_jsonl(SMALL / "qrels.jsonl")

    q_ids = [q["_id"] for q in queries]
    if len(q_ids) != len(set(q_ids)):
        raise RuntimeError("duplicate query ids")
    small_ids = [d["_id"] for d in small_docs]
    full_ids = [d["_id"] for d in full_docs]
    if len(small_ids) != len(set(small_ids)) or len(full_ids) != len(set(full_ids)):
        raise RuntimeError("duplicate corpus ids")
    if set(small_ids) & set(full_ids):
        raise RuntimeError("small and full corpus ids overlap; drowning pool is not disjoint")

    gold_map: dict[str, list[str]] = defaultdict(list)
    for rel in qrels:
        if int(rel["score"]) <= 0:
            continue
        gold_map[rel["query-id"]].append(rel["corpus-id"])
    if set(gold_map) != set(q_ids):
        raise RuntimeError("qrels query ids do not match queries")
    if any(len(v) != 2 for v in gold_map.values()):
        raise RuntimeError("expected exactly 2 gold documents per query")
    small_pos = {doc_id: i for i, doc_id in enumerate(small_ids)}
    gold_index = np.asarray(
        [[small_pos[doc_id] for doc_id in gold_map[qid]] for qid in q_ids],
        dtype=np.int32,
    )

    query_texts = [q["text"] for q in queries]
    attributes = [attribute_of(t) for t in query_texts]
    small_texts = [doc_text(d) for d in small_docs]
    full_texts = [doc_text(d) for d in full_docs]

    t_emb = time.perf_counter()
    query_emb = embed(model, query_texts, batch_size=256)
    embed_queries_s = time.perf_counter() - t_emb
    t_emb = time.perf_counter()
    small_emb = embed(model, small_texts, batch_size=256)
    embed_small_s = time.perf_counter() - t_emb
    t_emb = time.perf_counter()
    full_emb = embed(model, full_texts, batch_size=512)
    embed_full_s = time.perf_counter() - t_emb
    torch.cuda.synchronize()

    q_norm = np.linalg.norm(query_emb, axis=1)
    d_norm = np.linalg.norm(small_emb, axis=1)
    if not np.allclose(q_norm, 1.0, atol=1e-3) or not np.allclose(d_norm, 1.0, atol=1e-3):
        raise RuntimeError("embeddings are not L2-normalized")

    trunc_small = truncation_stats(model, small_texts)
    # Full corpus is the same template and length band; sample 2000 for the report.
    trunc_full = truncation_stats(model, full_texts[::25])

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(full_docs))
    if max(ADDED_SIZES) > len(full_docs):
        raise RuntimeError("added size exceeds full corpus")

    curve = []
    t_score = time.perf_counter()
    small_scores = cosine(query_emb, small_emb)
    small_key = id_key(small_ids)
    small_ranks = gold_ranks(small_scores, gold_index, small_key)
    small_recall = {str(k): pack_recall(recall_at_k(small_ranks, k)) for k in KS_SMALL}

    for n_added in ADDED_SIZES:
        subset = perm[:n_added]
        if n_added == 0:
            scores = small_scores
            ids = small_ids
            ranks = small_ranks
        else:
            doc_emb = np.concatenate([small_emb, full_emb[subset]], axis=0)
            ids = small_ids + [full_ids[i] for i in subset]
            scores = cosine(query_emb, doc_emb)
            ranks = gold_ranks(scores, gold_index, id_key(ids))
            del doc_emb
        row = {
            "added_documents": int(n_added),
            "total_documents": int(len(ids)),
            "recall": {str(k): pack_recall(recall_at_k(ranks, k)) for k in KS_CURVE},
        }
        curve.append(row)
        print(
            f"added={n_added} total={len(ids)} "
            f"R@2={row['recall']['2']['percent']:.2f} "
            f"R@10={row['recall']['10']['percent']:.2f}",
            flush=True,
        )
    scoring_s = time.perf_counter() - t_score

    # Diagnostic on the 46-document pool, for 200 seed-0 queries.
    diag_rng = np.random.default_rng(SEED)
    sample = np.sort(diag_rng.choice(len(queries), size=DIAG_N, replace=False))
    sample_scores = small_scores[sample]
    sample_gold = gold_index[sample]
    sample_ranks = small_ranks[sample]
    wrong_idx = top_wrong_index(sample_scores, sample_gold, small_key)
    wrong_ranks = gold_ranks(
        sample_scores, wrong_idx.reshape(-1, 1), small_key
    ).reshape(-1)

    small_words = [content_words(t) for t in small_texts]
    attr_words = [content_words(a) for a in attributes]

    records = []
    n_top1_gold = 0
    n_wrong_shares = 0
    n_top1_wrong = 0
    n_top1_wrong_and_shares = 0
    shared_counts = []
    nongold_share_rates = []
    better_ranks = []
    worse_ranks = []
    for row_i, qi in enumerate(sample):
        aw = attr_words[qi]
        golds = []
        for slot in range(2):
            doc_i = int(sample_gold[row_i, slot])
            golds.append(
                {
                    "corpus_id": small_ids[doc_i],
                    "rank": int(sample_ranks[row_i, slot]),
                    "cosine": round(float(sample_scores[row_i, doc_i]), 4),
                }
            )
        golds_sorted = sorted(golds, key=lambda g: (g["rank"], g["corpus_id"]))
        w = int(wrong_idx[row_i])
        shared = sorted(aw & small_words[w])
        top1_is_gold = int(sample_ranks[row_i].min()) == 1
        shares = bool(shared)
        if top1_is_gold:
            n_top1_gold += 1
        else:
            n_top1_wrong += 1
            if shares:
                n_top1_wrong_and_shares += 1
        if shares:
            n_wrong_shares += 1
        shared_counts.append(len(shared))
        gold_set = set(int(x) for x in sample_gold[row_i])
        nongold = [j for j in range(len(small_ids)) if j not in gold_set]
        nongold_share_rates.append(
            float(np.mean([bool(aw & small_words[j]) for j in nongold]))
        )
        better_ranks.append(golds_sorted[0]["rank"])
        worse_ranks.append(golds_sorted[1]["rank"])
        records.append(
            {
                "query_id": q_ids[qi],
                "attribute": attributes[qi],
                "attribute_content_words": sorted(aw),
                "gold": golds_sorted,
                "top_wrong": {
                    "corpus_id": small_ids[w],
                    "rank": int(wrong_ranks[row_i]),
                    "cosine": round(float(sample_scores[row_i, w]), 4),
                    "shares_content_word": shares,
                    "shared_words": shared,
                },
            }
        )

    better = np.asarray(better_ranks)
    worse = np.asarray(worse_ranks)
    both_in_top2 = float(np.mean((better <= 2) & (worse <= 2)))

    def rank_hist(arr: np.ndarray) -> dict:
        return {
            "1": int((arr == 1).sum()),
            "2": int((arr == 2).sum()),
            "3-5": int(((arr >= 3) & (arr <= 5)).sum()),
            "6-10": int(((arr >= 6) & (arr <= 10)).sum()),
            "11-20": int(((arr >= 11) & (arr <= 20)).sum()),
            "21+": int((arr >= 21).sum()),
        }

    # Same 200 queries against the full drowned pool, summary only.
    full_subset_emb = np.concatenate([small_emb, full_emb], axis=0)
    full_ids_all = small_ids + full_ids
    full_scores = cosine(query_emb[sample], full_subset_emb)
    full_key = id_key(full_ids_all)
    full_gold_ranks = gold_ranks(full_scores, sample_gold, full_key)
    full_wrong = top_wrong_index(full_scores, sample_gold, full_key)
    full_words = small_words + [content_words(t) for t in full_texts]
    full_share = 0
    full_top1_wrong = 0
    full_top1_wrong_and_shares = 0
    for row_i, qi in enumerate(sample):
        w = int(full_wrong[row_i])
        shares = bool(attr_words[qi] & full_words[w])
        top1_wrong = int(full_gold_ranks[row_i].min()) != 1
        if shares:
            full_share += 1
        if top1_wrong:
            full_top1_wrong += 1
            if shares:
                full_top1_wrong_and_shares += 1

    # Dataset fact: do gold documents contain the attribute's content words?
    gold_has_word = 0
    gold_with_words = 0
    empty_attr_queries = []
    nongold_share_all = []
    for qi, qid in enumerate(q_ids):
        aw = attr_words[qi]
        gset = set(int(x) for x in gold_index[qi])
        if not aw:
            empty_attr_queries.append(attributes[qi])
        for g in gset:
            if not aw:
                continue
            gold_with_words += 1
            if aw <= small_words[g]:
                gold_has_word += 1
        if not aw:
            continue
        nongold_share_all.append(
            float(
                np.mean(
                    [bool(aw & small_words[j]) for j in range(len(small_ids)) if j not in gset]
                )
            )
        )

    overlap_scores = np.zeros((len(queries), len(small_ids)), dtype=np.float32)
    for qi, aw in enumerate(attr_words):
        for j, words in enumerate(small_words):
            overlap_scores[qi, j] = float(len(aw & words))
    overlap_ranks = gold_ranks(overlap_scores, gold_index, small_key)
    lexical_baseline = {
        "description": (
            "Rank the 46 documents by how many attribute content words they share. "
            "Equal counts break by ascending document id. This is a CPU check, included in wall time."
        ),
        "n_documents": len(small_docs),
        "n_queries": len(queries),
        "recall": {str(k): pack_recall(recall_at_k(overlap_ranks, k)) for k in KS_SMALL},
        "fraction_both_golds_in_top2": round(float((overlap_ranks <= 2).all(axis=1).mean()), 6),
    }

    wall_s = time.perf_counter() - t0
    results = {
        "model": MODEL_NAME,
        "embedding_dim": dim,
        "similarity": "cosine",
        "device": {
            "cuda_visible_devices": "1",
            "torch_device": "cuda:0",
            "name": device_name,
            "capability": list(capability),
        },
        "software": {
            "torch": torch.__version__,
            "sentence_transformers": __import__("sentence_transformers").__version__,
            "numpy": np.__version__,
        },
        "wall_time_seconds": round(wall_s, 3),
        "timings_seconds": {
            "model_load": round(model_load_s, 3),
            "embed_queries": round(embed_queries_s, 3),
            "embed_small_documents": round(embed_small_s, 3),
            "embed_full_documents": round(embed_full_s, 3),
            "scoring_and_metrics": round(scoring_s, 3),
            "total": round(wall_s, 3),
        },
        "protocol": {
            "queries": "data/limit-small/queries.jsonl (identical to data/limit/queries.jsonl)",
            "relevance": "data/limit-small/qrels.jsonl, exactly 2 gold documents per query",
            "recall": (
                "macro-average over queries of |gold ∩ top-k| / |gold|. "
                "percent = 100 * fraction."
            ),
            "tie_break": "higher cosine first; equal cosine broken by ascending document id",
            "drowning": (
                "corpus = all 46 limit-small documents, followed by a nested prefix of "
                "numpy Generator(seed=0).permutation of the 50000 full-corpus rows. "
                "Added sizes 0, 100, 1000, 10000, 50000."
            ),
            "document_text": "title + text, titles are empty so this is the text field",
            "query_text": "raw query text, no instruction prefix",
            "tf32": False,
        },
        "truncation": {"small_corpus": trunc_small, "full_corpus_every_25th": trunc_full},
        "limit_small": {
            "n_documents": len(small_docs),
            "n_queries": len(queries),
            "recall": small_recall,
        },
        "drowning_curve": curve,
        "lexical_content_word_baseline": lexical_baseline,
        "dataset_facts": {
            "gold_documents_containing_all_attribute_content_words": {
                "n": gold_has_word,
                "of": gold_with_words,
                "fraction": round(gold_has_word / gold_with_words, 6),
            },
            "attributes_with_no_content_word": empty_attr_queries,
            "mean_fraction_of_nongold_small_docs_sharing_an_attribute_content_word": round(
                float(np.mean(nongold_share_all)), 6
            ),
            "note": (
                "Content-word containment is counted only when the attribute has at least one "
                "alphanumeric token longer than 3 characters. Attributes with no such token "
                "are listed separately. Empty attribute word sets do not count as a match."
            ),
        },
        "diagnostic": {
            "corpus": "limit-small (46 documents)",
            "n_queries": DIAG_N,
            "seed": SEED,
            "selection": "numpy Generator(0).choice without replacement, then sorted",
            "content_word": (
                "alphanumeric token of length > 3, lowercased. "
                "Compared between the query attribute (text inside 'Who likes ...?') "
                "and the top-ranked non-gold document."
            ),
            "summary": {
                "recall_at_2": pack_recall(recall_at_k(sample_ranks, 2)),
                "recall_at_10": pack_recall(recall_at_k(sample_ranks, 10)),
                "fraction_top1_is_gold": round(n_top1_gold / DIAG_N, 6),
                "fraction_both_golds_in_top2": round(both_in_top2, 6),
                "better_gold_rank": {
                    "mean": round(float(better.mean()), 4),
                    "median": round(float(np.median(better)), 4),
                    "histogram": rank_hist(better),
                },
                "worse_gold_rank": {
                    "mean": round(float(worse.mean()), 4),
                    "median": round(float(np.median(worse)), 4),
                    "histogram": rank_hist(worse),
                },
                "fraction_top_wrong_shares_content_word": round(n_wrong_shares / DIAG_N, 6),
                "mean_shared_content_words_with_top_wrong": round(float(np.mean(shared_counts)), 4),
                "mean_fraction_nongold_docs_sharing_a_content_word": round(
                    float(np.mean(nongold_share_rates)), 6
                ),
                "n_top1_wrong": n_top1_wrong,
                "fraction_top1_wrong_whose_top_wrong_shares_content_word": (
                    round(n_top1_wrong_and_shares / n_top1_wrong, 6) if n_top1_wrong else None
                ),
            },
            "queries": records,
        },
        "diagnostic_full_pool_summary": {
            "corpus": "46 limit-small documents plus all 50000 full-corpus documents",
            "n_queries": DIAG_N,
            "same_query_sample": True,
            "recall_at_2": pack_recall(recall_at_k(full_gold_ranks, 2)),
            "recall_at_10": pack_recall(recall_at_k(full_gold_ranks, 10)),
            "fraction_top_wrong_shares_content_word": round(full_share / DIAG_N, 6),
            "n_top1_wrong": full_top1_wrong,
            "fraction_top1_wrong_whose_top_wrong_shares_content_word": (
                round(full_top1_wrong_and_shares / full_top1_wrong, 6) if full_top1_wrong else None
            ),
            "better_gold_rank_mean": round(float(full_gold_ranks.min(axis=1).mean()), 4),
            "better_gold_rank_median": round(float(np.median(full_gold_ranks.min(axis=1))), 4),
        },
    }

    results_path = OUT / "results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n")
    report = render_report(results)
    (OUT / "report.md").write_text(report)
    print(f"wrote {results_path}", flush=True)
    print(f"wall_s {wall_s:.2f} device {device_name}", flush=True)


def render_report(r: dict) -> str:
    small = r["limit_small"]["recall"]
    lines = []
    lines.append("# Zero-shot MiniLM on LIMIT")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(
        f"Model `{r['model']}` ({r['embedding_dim']}-d), cosine similarity, "
        f"no query prefix. Device: CUDA_VISIBLE_DEVICES=1, seen as cuda:0, "
        f"{r['device']['name']} (capability {tuple(r['device']['capability'])}). "
        f"Torch {r['software']['torch']}, sentence-transformers {r['software']['sentence_transformers']}."
    )
    lines.append("")
    lines.append(
        f"Wall time {r['wall_time_seconds']:.1f}s "
        f"(model load {r['timings_seconds']['model_load']:.1f}s, "
        f"embed 1000 queries {r['timings_seconds']['embed_queries']:.1f}s, "
        f"embed 46 docs {r['timings_seconds']['embed_small_documents']:.1f}s, "
        f"embed 50000 docs {r['timings_seconds']['embed_full_documents']:.1f}s, "
        f"scoring {r['timings_seconds']['scoring_and_metrics']:.1f}s)."
    )
    lines.append("")
    lines.append(
        "Relevance is the limit-small qrels: 1000 queries, exactly 2 gold documents each, "
        "all inside the 46-document pool. Recall@k is the macro-average of "
        "|gold ∩ top-k| / 2, reported on a 0–100 scale. "
        "Equal cosines break toward the alphabetically earlier document id. "
        "TF32 was disabled for the similarity product."
    )
    lines.append("")
    trunc = r["truncation"]["small_corpus"]
    lines.append(
        f"MiniLM truncates at {trunc['max_seq_length']} tokens. "
        f"On the 46-document pool, token length min/median/max = "
        f"{trunc['min']}/{trunc['median']}/{trunc['max']}, "
        f"truncated {trunc['n_truncated']}/{trunc['n']}. "
        f"On every 25th full-corpus document, truncated "
        f"{r['truncation']['full_corpus_every_25th']['n_truncated']}/"
        f"{r['truncation']['full_corpus_every_25th']['n']} "
        f"(max tokens {r['truncation']['full_corpus_every_25th']['max']})."
    )
    lines.append("")
    lines.append("## limit-small (46 documents)")
    lines.append("")
    lines.append("| k | Recall@k |")
    lines.append("| ---: | ---: |")
    for k in ("2", "10", "20"):
        lines.append(f"| {k} | {small[k]['percent']:.2f} |")
    lines.append("")
    lines.append("## Drowning curve")
    lines.append("")
    lines.append(
        "The 46 relevant-pool documents stay in the index. Added documents are a nested "
        "prefix of one `numpy.random.default_rng(0).permutation` of the 50,000 full-corpus "
        "rows (seed 0). The last point ranks all 50,046 documents exactly; no approximate search."
    )
    lines.append("")
    lines.append("| Added | Total docs | Recall@2 | Recall@10 |")
    lines.append("| ---: | ---: | ---: | ---: |")
    for row in r["drowning_curve"]:
        lines.append(
            f"| {row['added_documents']} | {row['total_documents']} | "
            f"{row['recall']['2']['percent']:.2f} | {row['recall']['10']['percent']:.2f} |"
        )
    lines.append("")
    lines.append("## Comparison with the paper's large embedders")
    lines.append("")
    r2 = small["2"]["percent"]
    r10 = small["10"]["percent"]
    r20 = small["20"]["percent"]
    full_row = r["drowning_curve"][-1]
    lex = r["lexical_content_word_baseline"]["recall"]
    both_lex = r["lexical_content_word_baseline"]["fraction_both_golds_in_top2"]
    lines.append(
        f"On the 46-document pool MiniLM scores Recall@2 = {r2:.2f}, "
        f"Recall@10 = {r10:.2f}, Recall@20 = {r20:.2f}. "
        f"A ranker that only counts shared attribute content words, with the same id tie-break, "
        f"scores Recall@2 = {lex['2']['percent']:.2f}, Recall@10 = {lex['10']['percent']:.2f}, "
        f"Recall@20 = {lex['20']['percent']:.2f} on these same qrels "
        f"(both golds in the top 2 for {100 * both_lex:.1f}% of queries). "
        "That sits next to the paper's BM25 figure of about 98. A 4096-d embedder "
        "(Promptriever) is about 54. MiniLM is in the same direction as those dense models, "
        "and further from lexical retrieval: on a 46-document corpus the two gold documents "
        "are easy for word overlap and still usually missed by this embedder. "
        f"Random Recall@2 here would be 2/46 = 4.35, so {r2:.2f} is above chance and still a failure."
    )
    lines.append("")
    lines.append(
        f"Adding irrelevant-pool documents drops MiniLM from Recall@2 = {r2:.2f} "
        f"at 46 documents to Recall@2 = {full_row['recall']['2']['percent']:.2f} "
        f"and Recall@10 = {full_row['recall']['10']['percent']:.2f} at {full_row['total_documents']} documents. "
        "The paper's dense models on the official 50k corpus are about Recall@2 = 3 and "
        "Recall@100 = 19, while BM25 stays near 86. This curve is not that official qrel file: "
        "the 50k corpus here uses different document ids, and its qrels do not mark the same "
        "46 people. It is the drowning test that was asked for, and it moves the same way: "
        "extra documents collapse embedding recall while the labeled golds stay fixed."
    )
    lines.append("")
    lines.append("## Error analysis")
    lines.append("")
    d = r["diagnostic"]["summary"]
    facts = r["dataset_facts"]
    lines.append(
        "Sample: 200 queries from `numpy.random.default_rng(0).choice` (then sorted), "
        "ranked inside the 46-document pool. A content word is an alphanumeric token "
        "longer than 3 characters. The attribute is the span inside `Who likes ...?`. "
        "The top wrong document is the highest-ranked document that is not one of the two golds."
    )
    lines.append("")
    lines.append(
        f"On this sample, Recall@2 = {d['recall_at_2']['percent']:.2f} and "
        f"Recall@10 = {d['recall_at_10']['percent']:.2f}. "
        f"Top-1 is a gold document for {100 * d['fraction_top1_is_gold']:.1f}% of queries. "
        f"Both golds are in the top 2 for {100 * d['fraction_both_golds_in_top2']:.1f}%. "
        f"The better gold has mean rank {d['better_gold_rank']['mean']:.2f} "
        f"(median {d['better_gold_rank']['median']:.1f}); "
        f"the worse gold has mean rank {d['worse_gold_rank']['mean']:.2f} "
        f"(median {d['worse_gold_rank']['median']:.1f})."
    )
    lines.append("")
    lines.append(
        "Better-gold rank histogram (of 200): "
        + ", ".join(f"{k}={v}" for k, v in d["better_gold_rank"]["histogram"].items())
        + ". Worse-gold rank histogram: "
        + ", ".join(f"{k}={v}" for k, v in d["worse_gold_rank"]["histogram"].items())
        + "."
    )
    lines.append("")
    lines.append(
        f"The top wrong document shares an attribute content word on only "
        f"{100 * d['fraction_top_wrong_shares_content_word']:.1f}% of the 200 queries "
        f"(mean {d['mean_shared_content_words_with_top_wrong']:.2f} shared words). "
        f"A typical non-gold document in the 46 shares a content word "
        f"{100 * d['mean_fraction_nongold_docs_sharing_a_content_word']:.1f}% of the time, "
        f"so lexical overlap is only mildly enriched. "
        f"Among the {d['n_top1_wrong']} queries whose top-1 is already wrong, "
        f"{100 * d['fraction_top1_wrong_whose_top_wrong_shares_content_word']:.1f}% "
        f"have that overlap. The usual mistake is a biography that does not repeat the attribute."
    )
    lines.append("")
    gold_frac = facts["gold_documents_containing_all_attribute_content_words"]
    empty_attrs = ", ".join(facts["attributes_with_no_content_word"])
    n_attr_queries = r["limit_small"]["n_queries"] - len(facts["attributes_with_no_content_word"])
    lines.append(
        f"Dataset check: {gold_frac['n']}/{gold_frac['of']} gold documents whose attribute has a "
        f"content word contain every such word ({100 * gold_frac['fraction']:.1f}%). "
        f"Attributes with no alphanumeric token longer than 3 characters are: {empty_attrs}. "
        f"Across the {n_attr_queries} queries whose attribute has a content word, a non-gold document in the 46 shares such a word "
        f"{100 * facts['mean_fraction_of_nongold_small_docs_sharing_an_attribute_content_word']:.1f}% "
        f"of the time. The gold text states the attribute, word overlap recovers it "
        f"(Recall@2 = {r['lexical_content_word_baseline']['recall']['2']['percent']:.2f}), "
        f"and MiniLM still does not put both golds first. The embedding is not tracking the "
        f"queried attribute inside an otherwise similar biography."
    )
    lines.append("")
    full_d = r["diagnostic_full_pool_summary"]
    lines.append(
        f"The same 200 queries on the 50,046-document pool: "
        f"Recall@2 = {full_d['recall_at_2']['percent']:.2f}, "
        f"Recall@10 = {full_d['recall_at_10']['percent']:.2f}, "
        f"mean rank of the better gold = {full_d['better_gold_rank_mean']:.1f} "
        f"(median {full_d['better_gold_rank_median']:.1f}). "
        f"Top-1 is wrong on {full_d['n_top1_wrong']}/200 queries, and the top wrong document "
        f"shares an attribute content word on "
        f"{100 * full_d['fraction_top_wrong_shares_content_word']:.1f}% of queries "
        f"({100 * full_d['fraction_top1_wrong_whose_top_wrong_shares_content_word']:.1f}% "
        f"of the top-1 errors)."
    )
    lines.append("")
    # Concrete examples: up to 3 misses with overlap, 3 misses without.
    misses_share = []
    misses_clean = []
    for rec in r["diagnostic"]["queries"]:
        if rec["gold"][0]["rank"] == 1:
            continue
        if rec["top_wrong"]["shares_content_word"]:
            misses_share.append(rec)
        else:
            misses_clean.append(rec)
    lines.append("Examples where top-1 is not gold (limit-small ranks):")
    lines.append("")
    if misses_share:
        lines.append("Top wrong document shares an attribute content word:")
        lines.append("")
        for rec in misses_share[:3]:
            g = ", ".join(
                f"{x['corpus_id']}@{x['rank']} (cosine {x['cosine']:.3f})" for x in rec["gold"]
            )
            lines.append(
                f"- `{rec['query_id']}` attribute `{rec['attribute']}`: gold ranks {g}; "
                f"top wrong `{rec['top_wrong']['corpus_id']}` at rank {rec['top_wrong']['rank']} "
                f"(cosine {rec['top_wrong']['cosine']:.3f}) "
                f"sharing {rec['top_wrong']['shared_words']}."
            )
        lines.append("")
    if misses_clean:
        lines.append("Top wrong document shares no attribute content word:")
        lines.append("")
        for rec in misses_clean[:3]:
            g = ", ".join(
                f"{x['corpus_id']}@{x['rank']} (cosine {x['cosine']:.3f})" for x in rec["gold"]
            )
            lines.append(
                f"- `{rec['query_id']}` attribute `{rec['attribute']}`: gold ranks {g}; "
                f"top wrong `{rec['top_wrong']['corpus_id']}` at rank {rec['top_wrong']['rank']} "
                f"(cosine {rec['top_wrong']['cosine']:.3f})."
            )
        lines.append("")
    lines.append("Per-query ranks for all 200 sampled queries are in `results.json`.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
