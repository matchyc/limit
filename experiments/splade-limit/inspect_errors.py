#!/usr/bin/env python3
"""Print SPLADE top documents for official LIMIT queries that miss the golds."""

from __future__ import annotations

import json
import os
import re
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
MODEL_NAME = "naver/splade-cocondenser-ensembledistil"
DOC_BATCH = 64

sys.path.insert(0, str(SCHEME))
from run_scheme_failures import (  # noqa: E402
    attribute_of,
    doc_text,
    id_key,
    load_jsonl,
    split_likes,
)


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "to_dense"):
        value = value.to_dense()
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def likes_hits(attrs: list[str], needle: str) -> list[str]:
    hits = []
    low = needle.lower()
    for attr in attrs:
        if needle == attr or needle in attr or attr in needle or low in attr.lower():
            hits.append(attr)
    return hits


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

    from sentence_transformers import SparseEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SparseEncoder(MODEL_NAME, device=device)
    q_emb = model.encode(
        query_texts, batch_size=32, convert_to_tensor=True,
        convert_to_sparse_tensor=True, show_progress_bar=False,
    )
    scores = np.empty((len(query_texts), n_docs), dtype=np.float32)
    for start in range(0, n_docs, DOC_BATCH):
        end = min(start + DOC_BATCH, n_docs)
        d_emb = model.encode(
            doc_texts[start:end], batch_size=DOC_BATCH, convert_to_tensor=True,
            convert_to_sparse_tensor=True, show_progress_bar=False,
        )
        scores[:, start:end] = to_numpy(model.similarity(q_emb, d_emb))
        if end % 10000 == 0 or end == n_docs:
            print(f"encoded {end}/{n_docs}", flush=True)
        del d_emb
        torch.cuda.empty_cache()

    want = ["Honey", "Go", "Horses", "Joshua Trees", "Birch Trees", "Snow"]
    want_rows = []
    for i, attr in enumerate(query_attrs):
        if attr in want:
            want_rows.append(i)
    # Also the 8 worst better-gold ranks among all queries.
    better = []
    for row in range(len(query_texts)):
        order = np.lexsort((key, -scores[row]))
        rank_of = {int(d): r + 1 for r, d in enumerate(order)}
        g_ranks = [rank_of[int(g)] for g in gold_index[row]]
        better.append(min(g_ranks))
    worst = np.argsort(-np.asarray(better))[:12]

    cases = []
    seen = set()
    for row in list(want_rows) + [int(i) for i in worst]:
        if row in seen:
            continue
        seen.add(row)
        attr = query_attrs[row]
        order = np.lexsort((key, -scores[row]))
        rank_of = {int(d): r + 1 for r, d in enumerate(order)}
        golds = gold_index[row].tolist()
        top = []
        for rank, di in enumerate(order[:8], start=1):
            di = int(di)
            name, attrs = parsed[di]
            hits = likes_hits(attrs, attr)
            top.append({
                "rank": rank,
                "id": doc_ids[di],
                "score": round(float(scores[row, di]), 4),
                "is-gold": di in golds,
                "matching-likes-items": hits[:8],
                "n-likes": len(attrs),
            })
        gold_rows = []
        for g in golds:
            name, attrs = parsed[int(g)]
            gold_rows.append({
                "rank": rank_of[int(g)],
                "id": doc_ids[int(g)],
                "score": round(float(scores[row, int(g)]), 4),
                "matching-likes-items": likes_hits(attrs, attr),
            })
        n_exact = int(sum(attr in parsed[int(di)][1] for di in order[:10]))
        n_sub = int(sum(
            any(attr in a or a.find(attr) >= 0 for a in parsed[int(di)][1])
            for di in order[:10]
        ))
        cases.append({
            "query-id": query_ids[row],
            "query": query_texts[row],
            "attribute": attr,
            "gold-ranks": [rank_of[int(g)] for g in golds],
            "top8": top,
            "golds": gold_rows,
            "top10-with-exact-attribute": n_exact,
            "top10-with-attribute-as-substring": n_sub,
        })
        print(json.dumps(cases[-1], ensure_ascii=False, indent=2), flush=True)

    out = {
        "model": MODEL_NAME,
        "n-queries-better-gold-rank-gt-2": int(sum(b > 2 for b in better)),
        "n-queries-better-gold-rank-gt-10": int(sum(b > 10 for b in better)),
        "cases": cases,
    }
    (HERE / "error_cases.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print("wrote error_cases.json", flush=True)


if __name__ == "__main__":
    main()
