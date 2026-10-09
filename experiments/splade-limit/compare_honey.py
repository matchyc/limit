#!/usr/bin/env python3
"""Top documents for 'Who likes Honey?' under BM25, MiniLM, and SPLADE."""

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
MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DOC_BATCH = 64
QID = "query_730"

sys.path.insert(0, str(SCHEME))
from run_scheme_failures import (  # noqa: E402
    BM25,
    doc_text,
    encode_sentences,
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


def pack_top(scores: np.ndarray, row: int, parsed, doc_ids, golds: set[int], key, attr: str, k: int = 10):
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
    for g in sorted(golds):
        _name, attrs = parsed[g]
        gold_rows.append({
            "rank": rank_of[g],
            "id": doc_ids[g],
            "score": round(float(scores[row, g]), 4),
            "matching-likes-items": likes_hits(attrs, attr),
        })
    return top, gold_rows


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "to_dense"):
        value = value.to_dense()
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


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
    row = query_ids.index(QID)
    attr = "Honey"
    id_to_pos = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    golds = {id_to_pos[g] for g in gold_map[QID]}
    key = id_key(doc_ids)
    n_docs = len(doc_ids)
    print(f"query={query_texts[row]!r} golds={[doc_ids[g] for g in golds]}", flush=True)

    print("bm25", flush=True)
    bm25_scores = BM25(doc_texts, doc_ids).score([query_texts[row]]).astype(np.float32)
    bm25_top, bm25_gold = pack_top(bm25_scores, 0, parsed, doc_ids, golds, key, attr)
    n_honey_token = sum("honey" in t.lower() for t in doc_texts)
    print("bm25 golds", bm25_gold, "n_docs_with_honey_substr", n_honey_token, flush=True)

    from sentence_transformers import SentenceTransformer, SparseEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"minilm {device}", flush=True)
    minilm = SentenceTransformer(MINILM_NAME, device=device)
    q_vec = encode_sentences(minilm, [query_texts[row]])
    d_vec = encode_sentences(minilm, doc_texts, batch_size=256)
    dense_scores = q_vec @ d_vec.T
    dense_top, dense_gold = pack_top(dense_scores, 0, parsed, doc_ids, golds, key, attr)
    print("dense golds", dense_gold, flush=True)
    del minilm, q_vec, d_vec
    torch.cuda.empty_cache()

    print("splade", flush=True)
    splade = SparseEncoder(SPLADE_NAME, device=device)
    q_emb = splade.encode(
        [query_texts[row]], batch_size=1, convert_to_tensor=True,
        convert_to_sparse_tensor=True, show_progress_bar=False,
    )
    splade_scores = np.empty((1, n_docs), dtype=np.float32)
    for start in range(0, n_docs, DOC_BATCH):
        end = min(start + DOC_BATCH, n_docs)
        d_emb = splade.encode(
            doc_texts[start:end], batch_size=DOC_BATCH, convert_to_tensor=True,
            convert_to_sparse_tensor=True, show_progress_bar=False,
        )
        splade_scores[:, start:end] = to_numpy(splade.similarity(q_emb, d_emb))
        if end % 10000 == 0 or end == n_docs:
            print(f"  {end}/{n_docs}", flush=True)
        del d_emb
        torch.cuda.empty_cache()
    splade_top, splade_gold = pack_top(splade_scores, 0, parsed, doc_ids, golds, key, attr)
    print("splade golds", splade_gold, flush=True)

    out = {
        "query-id": QID,
        "query": query_texts[row],
        "golds": [doc_ids[g] for g in sorted(golds)],
        "bm25": {"top10": bm25_top, "golds": bm25_gold},
        "minilm-whole": {"top10": dense_top, "golds": dense_gold},
        "splade-whole": {"top10": splade_top, "golds": splade_gold},
    }
    path = HERE / "honey_compare.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
