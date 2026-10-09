#!/usr/bin/env python3
"""Queries where BM25 puts both golds in the top 2 and SPLADE does not."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FULL = ROOT / "data" / "limit"
SCHEME = ROOT / "experiments" / "scheme-failures"
SPLADE_NAME = "naver/splade-cocondenser-ensembledistil"
DOC_BATCH = 64

sys.path.insert(0, str(SCHEME))
from run_scheme_failures import (  # noqa: E402
    BM25,
    attribute_of,
    doc_text,
    gold_ranks,
    id_key,
    load_jsonl,
    split_likes,
)


def likes_hits(attrs: list[str], needle: str) -> list[str]:
    hits = []
    for attr in attrs:
        if needle == attr or needle in attr or attr in needle:
            hits.append(attr)
    return hits


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "to_dense"):
        value = value.to_dense()
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def pack_top(scores, row, parsed, doc_ids, golds, key, attr, k=8):
    order = np.lexsort((key, -scores[row]))
    rank_of = {int(d): r + 1 for r, d in enumerate(order)}
    top = []
    for rank, di in enumerate(order[:k], start=1):
        di = int(di)
        _name, attrs = parsed[di]
        top.append({
            "rank": rank,
            "id": doc_ids[di],
            "score": round(float(scores[row, di]), 4),
            "is-gold": di in golds,
            "matching-likes-items": likes_hits(attrs, attr),
        })
    gold_rows = []
    for g in golds:
        _name, attrs = parsed[int(g)]
        gold_rows.append({
            "rank": rank_of[int(g)],
            "id": doc_ids[int(g)],
            "score": round(float(scores[row, int(g)]), 4),
            "matching-likes-items": likes_hits(attrs, attr),
        })
    gold_rows.sort(key=lambda x: x["rank"])
    return top, gold_rows


def main() -> None:
    docs = load_jsonl(FULL / "corpus.jsonl")
    queries = load_jsonl(FULL / "queries.jsonl")
    qrels = load_jsonl(FULL / "qrels.jsonl")
    gold_map: dict[str, list[str]] = {}
    for row in qrels:
        gold_map.setdefault(row["query-id"], []).append(row["corpus-id"])
    doc_ids = [row["_id"] for row in docs]
    doc_texts = [doc_text(row) for row in docs]
    parsed = [split_likes(text) for text in doc_texts]
    query_ids = [row["_id"] for row in queries]
    query_texts = [row["text"] for row in queries]
    query_attrs = [attribute_of(text) for text in query_texts]
    id_to_pos = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    gold_index = np.asarray(
        [[id_to_pos[g] for g in gold_map[qid]] for qid in query_ids], dtype=np.int32
    )
    key = id_key(doc_ids)
    n_docs = len(doc_ids)
    n_queries = len(query_texts)

    print("bm25", flush=True)
    bm25_scores = BM25(doc_texts, doc_ids).score(query_texts).astype(np.float32)
    bm25_ranks = gold_ranks(bm25_scores, gold_index, key)

    from sentence_transformers import SparseEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"splade {device}", flush=True)
    model = SparseEncoder(SPLADE_NAME, device=device)
    q_emb = model.encode(
        query_texts, batch_size=32, convert_to_tensor=True,
        convert_to_sparse_tensor=True, show_progress_bar=False,
    )
    splade_scores = np.empty((n_queries, n_docs), dtype=np.float32)
    for start in range(0, n_docs, DOC_BATCH):
        end = min(start + DOC_BATCH, n_docs)
        d_emb = model.encode(
            doc_texts[start:end], batch_size=DOC_BATCH, convert_to_tensor=True,
            convert_to_sparse_tensor=True, show_progress_bar=False,
        )
        splade_scores[:, start:end] = to_numpy(model.similarity(q_emb, d_emb))
        if end % 10000 == 0 or end == n_docs:
            print(f"  {end}/{n_docs}", flush=True)
        del d_emb
        torch.cuda.empty_cache()
    splade_ranks = gold_ranks(splade_scores, gold_index, key)

    bm25_both = (bm25_ranks <= 2).all(axis=1)
    splade_miss = ~((splade_ranks <= 2).all(axis=1))
    hits = np.flatnonzero(bm25_both & splade_miss)
    # Prefer cases where SPLADE is clearly wrong (better gold outside top 10).
    worse = []
    for row in hits.tolist():
        worse.append((int(splade_ranks[row].min()), query_attrs[row], row))
    worse.sort(reverse=True)
    print(
        f"bm25-both-in-top2={int(bm25_both.sum())} "
        f"splade-not-both={int(splade_miss.sum())} "
        f"bm25-ok-splade-bad={len(hits)}",
        flush=True,
    )

    pick = [row for _m, _a, row in worse[:8]]
    cases = []
    for row in pick:
        attr = query_attrs[row]
        golds = set(int(g) for g in gold_index[row])
        b_top, b_gold = pack_top(bm25_scores, row, parsed, doc_ids, golds, key, attr)
        s_top, s_gold = pack_top(splade_scores, row, parsed, doc_ids, golds, key, attr)
        case = {
            "query-id": query_ids[row],
            "query": query_texts[row],
            "attribute": attr,
            "bm25-gold-ranks": bm25_ranks[row].tolist(),
            "splade-gold-ranks": splade_ranks[row].tolist(),
            "bm25-top8": b_top,
            "splade-top8": s_top,
            "bm25-golds": b_gold,
            "splade-golds": s_gold,
        }
        cases.append(case)
        print(json.dumps({
            "query": case["query"],
            "bm25-gold-ranks": case["bm25-gold-ranks"],
            "splade-gold-ranks": case["splade-gold-ranks"],
            "splade-top8": [
                {"rank": x["rank"], "id": x["id"], "is-gold": x["is-gold"],
                 "hits": x["matching-likes-items"]}
                for x in s_top
            ],
        }, ensure_ascii=False, indent=2), flush=True)

    out = {
        "n-queries": n_queries,
        "bm25-both-in-top2": int(bm25_both.sum()),
        "splade-not-both-in-top2": int(splade_miss.sum()),
        "bm25-ok-and-splade-bad": int(len(hits)),
        "cases": cases,
    }
    path = HERE / "bm25_ok_splade_bad.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
