#!/usr/bin/env python3
"""SPLADE++ on the official LIMIT queries (original 'Who likes X?').

limit-small (46 docs) numbers are copied from experiments/splade-gloss.
This script scores the 50k-document LIMIT split, which is a different draw.

Run:
  CUDA_VISIBLE_DEVICES=1 ../zero-shot-embed/.venv/bin/python run_splade_limit.py
"""

from __future__ import annotations

import json
import os
import sys
import time
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
SMALL_RESULTS = ROOT / "experiments" / "splade-gloss" / "results.json"
MODEL_NAME = "naver/splade-cocondenser-ensembledistil"
DOC_BATCH = 64

sys.path.insert(0, str(SCHEME))
from run_scheme_failures import (  # noqa: E402
    BM25,
    attribute_of,
    doc_text,
    gold_ranks,
    id_key,
    load_jsonl,
)


def pack(ranks: np.ndarray) -> dict:
    n_queries, n_gold = ranks.shape
    out = {}
    for k in (2, 10, 20, 100):
        hits = int((ranks <= k).sum())
        out[f"recall@{k}"] = round(100.0 * hits / (n_queries * n_gold), 2)
    out["both-in-top2"] = round(100.0 * float((ranks <= 2).all(axis=1).mean()), 2)
    better = ranks.min(axis=1)
    out["mean-better-gold-rank"] = round(float(better.mean()), 2)
    out["median-better-gold-rank"] = round(float(np.median(better)), 2)
    return out


def evaluate(scores: np.ndarray, gold_index: np.ndarray, key: np.ndarray) -> dict:
    return pack(gold_ranks(scores, gold_index, key))


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "to_dense"):
        value = value.to_dense()
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def main() -> None:
    t0 = time.time()
    docs = load_jsonl(FULL / "corpus.jsonl")
    queries = load_jsonl(FULL / "queries.jsonl")
    qrels = load_jsonl(FULL / "qrels.jsonl")
    gold_map: dict[str, list[str]] = {}
    for row in qrels:
        gold_map.setdefault(row["query-id"], []).append(row["corpus-id"])

    doc_ids = [row["_id"] for row in docs]
    doc_texts = [doc_text(row) for row in docs]
    query_ids = [row["_id"] for row in queries]
    query_texts = [row["text"] for row in queries]
    query_attrs = [attribute_of(text) for text in query_texts]
    id_to_pos = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    gold_index = []
    for qid in query_ids:
        golds = gold_map[qid]
        if len(golds) != 2:
            raise SystemExit(f"bad golds for {qid}")
        gold_index.append([id_to_pos[g] for g in golds])
    gold_index = np.asarray(gold_index, dtype=np.int32)
    key = id_key(doc_ids)
    n_docs = len(doc_ids)
    print(f"full docs={n_docs} queries={len(query_texts)} unique_attrs={len(set(query_attrs))}", flush=True)

    print("bm25", flush=True)
    bm25_scores = BM25(doc_texts, doc_ids).score(query_texts).astype(np.float32)
    bm25_metrics = evaluate(bm25_scores, gold_index, key)
    print("bm25", bm25_metrics, flush=True)
    del bm25_scores

    from sentence_transformers import SparseEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {MODEL_NAME} on {device}", flush=True)
    model = SparseEncoder(MODEL_NAME, device=device)
    tokenizer = model.tokenizer
    sample_n = min(200, n_docs)
    sample_tok = tokenizer(doc_texts[:sample_n], add_special_tokens=True, truncation=False)
    n_tokens = [len(ids) for ids in sample_tok["input_ids"]]
    print(
        f"sample {sample_n} doc tokens min={min(n_tokens)} median={int(np.median(n_tokens))} "
        f"max={max(n_tokens)} model_max={tokenizer.model_max_length}",
        flush=True,
    )

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
        block = model.similarity(q_emb, d_emb)
        scores[:, start:end] = to_numpy(block)
        if start == 0 or end == n_docs or end % 3200 == 0:
            print(f"encoded docs {end}/{n_docs}", flush=True)
        del d_emb, block
        torch.cuda.empty_cache()

    splade_metrics = evaluate(scores, gold_index, key)
    print("splade", splade_metrics, flush=True)

    small = json.loads(SMALL_RESULTS.read_text()) if SMALL_RESULTS.exists() else None
    results = {
        "model": MODEL_NAME,
        "device": device,
        "seconds": round(time.time() - t0, 1),
        "limit": {
            "docs": n_docs,
            "queries": len(query_texts),
            "bm25": bm25_metrics,
            "splade-whole": splade_metrics,
        },
        "limit-small-from-splade-gloss": {
            "docs": small["docs"] if small else None,
            "queries": small["queries"] if small else None,
            "bm25": small["original"]["bm25"] if small else None,
            "splade-whole": small["original"]["splade-whole"] if small else None,
            "splade-chunk-max": small["original"]["splade-chunk-max"] if small else None,
        },
        "paper": {
            "bm25-small": {"recall@2": 97.8, "recall@10": 100.0},
            "bm25-full": {"recall@2": 85.7, "recall@10": 90.4, "recall@100": 93.6},
            "gte-moderncolbert-small": {"recall@2": 83.5, "recall@10": 97.6},
            "gte-moderncolbert-full": {"recall@2": 23.1, "recall@10": 34.6, "recall@100": 54.8},
            "promptriever-4096-full": {"recall@2": 3.0, "recall@10": 6.8, "recall@100": 18.9},
        },
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    write_report(results, HERE / "report.md")
    print(f"wrote results in {results['seconds']}s", flush=True)


def write_report(res: dict, path: Path) -> None:
    small = res["limit-small-from-splade-gloss"]
    full = res["limit"]
    paper = res["paper"]
    lines = [
        "# SPLADE++ on official LIMIT queries",
        "",
        f"Model `{res['model']}` on {res['device']}. Original `Who likes X?` queries. "
        f"limit and limit-small are different draws. {res['seconds']}s.",
        "",
        "Recall is macro over the two golds, in percent.",
        "",
        "## limit-small (46 documents)",
        "",
        "| score | R@2 | R@10 | R@20 | both@2 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| BM25 (ours) | {small['bm25']['recall@2']} | {small['bm25']['recall@10']} | {small['bm25']['recall@20']} | {small['bm25']['both-in-top2']} |",
        f"| SPLADE whole | {small['splade-whole']['recall@2']} | {small['splade-whole']['recall@10']} | {small['splade-whole']['recall@20']} | {small['splade-whole']['both-in-top2']} |",
        f"| SPLADE chunk-max | {small['splade-chunk-max']['recall@2']} | {small['splade-chunk-max']['recall@10']} | {small['splade-chunk-max']['recall@20']} | {small['splade-chunk-max']['both-in-top2']} |",
        f"| paper BM25 | {paper['bm25-small']['recall@2']} | {paper['bm25-small']['recall@10']} | — | — |",
        f"| paper GTE-ModernColBERT | {paper['gte-moderncolbert-small']['recall@2']} | {paper['gte-moderncolbert-small']['recall@10']} | — | — |",
        "",
        "## limit (50k documents)",
        "",
        "| score | R@2 | R@10 | R@20 | R@100 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| BM25 (ours) | {full['bm25']['recall@2']} | {full['bm25']['recall@10']} | {full['bm25']['recall@20']} | {full['bm25']['recall@100']} |",
        f"| SPLADE whole | {full['splade-whole']['recall@2']} | {full['splade-whole']['recall@10']} | {full['splade-whole']['recall@20']} | {full['splade-whole']['recall@100']} |",
        f"| paper BM25 | {paper['bm25-full']['recall@2']} | {paper['bm25-full']['recall@10']} | — | {paper['bm25-full']['recall@100']} |",
        f"| paper GTE-ModernColBERT | {paper['gte-moderncolbert-full']['recall@2']} | {paper['gte-moderncolbert-full']['recall@10']} | — | {paper['gte-moderncolbert-full']['recall@100']} |",
        f"| paper Promptriever 4096 | {paper['promptriever-4096-full']['recall@2']} | {paper['promptriever-4096-full']['recall@10']} | — | {paper['promptriever-4096-full']['recall@100']} |",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
