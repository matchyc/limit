#!/usr/bin/env python3
"""SPLADE-v3 on official LIMIT queries (original 'Who likes X?').

naver/splade-v3 (BERT) is Hugging Face gated without a token. This run uses the
public v3 checkpoint naver/splade-v3-distilbert from the same paper
(Lassance et al., 2024, arXiv:2403.06789). Architecture is still MLM + max-pool
SPLADE; DistilBERT instead of CoCondenser-BERT.

Run:
  CUDA_VISIBLE_DEVICES=1 ../zero-shot-embed/.venv/bin/python run_splade_v3.py
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
SMALL = ROOT / "data" / "limit-small"
FULL = ROOT / "data" / "limit"
SCHEME = ROOT / "experiments" / "scheme-failures"
PLUSPLUS = ROOT / "experiments" / "splade-limit" / "results.json"
MODEL_NAME = "naver/splade-v3-distilbert"
DOC_BATCH = 64

sys.path.insert(0, str(SCHEME))
from run_scheme_failures import (  # noqa: E402
    BM25,
    attribute_of,
    doc_text,
    gold_ranks,
    group_chunks,
    id_key,
    load_jsonl,
    split_likes,
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


def load_split(path: Path) -> tuple[list[str], list[str], list[str], np.ndarray, np.ndarray]:
    docs = load_jsonl(path / "corpus.jsonl")
    queries = load_jsonl(path / "queries.jsonl")
    qrels = load_jsonl(path / "qrels.jsonl")
    gold_map: dict[str, list[str]] = {}
    for row in qrels:
        gold_map.setdefault(row["query-id"], []).append(row["corpus-id"])
    doc_ids = [row["_id"] for row in docs]
    doc_texts = [doc_text(row) for row in docs]
    query_texts = [row["text"] for row in queries]
    query_ids = [row["_id"] for row in queries]
    id_to_pos = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    gold_index = np.asarray(
        [[id_to_pos[g] for g in gold_map[qid]] for qid in query_ids], dtype=np.int32
    )
    return doc_ids, doc_texts, query_texts, gold_index, id_key(doc_ids)


def encode_scores(model, query_texts: list[str], doc_texts: list[str], batch: int) -> np.ndarray:
    q_emb = model.encode(
        query_texts, batch_size=32, convert_to_tensor=True,
        convert_to_sparse_tensor=True, show_progress_bar=False,
    )
    scores = np.empty((len(query_texts), len(doc_texts)), dtype=np.float32)
    for start in range(0, len(doc_texts), batch):
        end = min(start + batch, len(doc_texts))
        d_emb = model.encode(
            doc_texts[start:end], batch_size=batch, convert_to_tensor=True,
            convert_to_sparse_tensor=True, show_progress_bar=False,
        )
        scores[:, start:end] = to_numpy(model.similarity(q_emb, d_emb))
        if start == 0 or end == len(doc_texts) or end % 3200 == 0:
            print(f"  encoded docs {end}/{len(doc_texts)}", flush=True)
        del d_emb
        torch.cuda.empty_cache()
    return scores


def chunk_max_scores(model, query_texts: list[str], parsed) -> np.ndarray:
    chunk_texts, owners = group_chunks(parsed, 1)
    q_emb = model.encode(
        query_texts, batch_size=32, convert_to_tensor=True,
        convert_to_sparse_tensor=True, show_progress_bar=False,
    )
    c_emb = model.encode(
        chunk_texts, batch_size=32, convert_to_tensor=True,
        convert_to_sparse_tensor=True, show_progress_bar=False,
    )
    token = to_numpy(model.similarity(q_emb, c_emb))
    n_docs = int(owners.max()) + 1
    scores = np.full((len(query_texts), n_docs), np.float32(-1e9))
    for doc_index in range(n_docs):
        cols = owners == doc_index
        if np.any(cols):
            scores[:, doc_index] = token[:, cols].max(axis=1)
    return scores


def main() -> None:
    t0 = time.time()
    print(f"trying gated naver/splade-v3 then falling back to {MODEL_NAME}", flush=True)
    from sentence_transformers import SparseEncoder
    from huggingface_hub.errors import GatedRepoError

    device = "cuda" if torch.cuda.is_available() else "cpu"
    used = "naver/splade-v3"
    try:
        model = SparseEncoder(used, device=device)
        gated_ok = True
    except (GatedRepoError, OSError) as err:
        print(f"naver/splade-v3 unavailable: {type(err).__name__}: {str(err).splitlines()[0][:200]}", flush=True)
        used = MODEL_NAME
        gated_ok = False
        model = SparseEncoder(used, device=device)
    print(f"loaded {used} on {device} gated_ok={gated_ok}", flush=True)

    plusplus = json.loads(PLUSPLUS.read_text()) if PLUSPLUS.exists() else None

    print("limit-small", flush=True)
    small_ids, small_texts, small_q, small_gold, small_key = load_split(SMALL)
    small_attrs = [attribute_of(t) for t in small_q]
    print(
        f"  docs={len(small_texts)} queries={len(small_q)} unique_attrs={len(set(small_attrs))}",
        flush=True,
    )
    small_bm25 = evaluate(
        BM25(small_texts, small_ids).score(small_q).astype(np.float32),
        small_gold, small_key,
    )
    small_whole = evaluate(encode_scores(model, small_q, small_texts, 8), small_gold, small_key)
    small_parsed = [split_likes(t) for t in small_texts]
    small_chunk = evaluate(chunk_max_scores(model, small_q, small_parsed), small_gold, small_key)
    print("  bm25", small_bm25, flush=True)
    print("  whole", small_whole, flush=True)
    print("  chunk-max", small_chunk, flush=True)

    print("limit 50k", flush=True)
    full_ids, full_texts, full_q, full_gold, full_key = load_split(FULL)
    print(f"  docs={len(full_texts)} queries={len(full_q)}", flush=True)
    full_bm25 = evaluate(
        BM25(full_texts, full_ids).score(full_q).astype(np.float32),
        full_gold, full_key,
    )
    print("  bm25", full_bm25, flush=True)
    full_whole = evaluate(encode_scores(model, full_q, full_texts, DOC_BATCH), full_gold, full_key)
    print("  whole", full_whole, flush=True)

    results = {
        "requested-model": "naver/splade-v3",
        "used-model": used,
        "gated-bert-v3-available": gated_ok,
        "device": device,
        "seconds": round(time.time() - t0, 1),
        "limit-small": {
            "docs": len(small_texts),
            "queries": len(small_q),
            "bm25": small_bm25,
            "splade-whole": small_whole,
            "splade-chunk-max": small_chunk,
        },
        "limit": {
            "docs": len(full_texts),
            "queries": len(full_q),
            "bm25": full_bm25,
            "splade-whole": full_whole,
        },
        "splade++-ensembledistil": {
            "model": plusplus["model"] if plusplus else None,
            "limit-small-whole": plusplus["limit-small-from-splade-gloss"]["splade-whole"] if plusplus else None,
            "limit-whole": plusplus["limit"]["splade-whole"] if plusplus else None,
        },
        "paper": {
            "splade-v3-bert-msmarco-mrr@10": 40.2,
            "splade-v3-distilbert-msmarco-mrr@10": 38.7,
            "splade++-ensembledistil-msmarco-mrr@10": 38.0,
            "bm25-full": {"recall@2": 85.7, "recall@10": 90.4, "recall@100": 93.6},
            "gte-moderncolbert-full": {"recall@2": 23.1, "recall@10": 34.6, "recall@100": 54.8},
        },
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    write_report(results, HERE / "report.md")
    print(f"wrote results in {results['seconds']}s", flush=True)


def write_report(res: dict, path: Path) -> None:
    small = res["limit-small"]
    full = res["limit"]
    pp = res["splade++-ensembledistil"]
    paper = res["paper"]
    pp_small = pp["limit-small-whole"] or {}
    pp_full = pp["limit-whole"] or {}
    lines = [
        "# SPLADE-v3 on official LIMIT queries",
        "",
        f"Requested `naver/splade-v3`. Used `{res['used-model']}` on {res['device']}. "
        f"BERT v3 gated={not res['gated-bert-v3-available']}. {res['seconds']}s.",
        "",
        "Recall is macro over the two golds, in percent. Original `Who likes X?` queries.",
        "",
        "## limit-small (46 documents)",
        "",
        "| score | R@2 | R@10 | R@20 | both@2 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| BM25 | {small['bm25']['recall@2']} | {small['bm25']['recall@10']} | {small['bm25']['recall@20']} | {small['bm25']['both-in-top2']} |",
        f"| SPLADE-v3 whole | {small['splade-whole']['recall@2']} | {small['splade-whole']['recall@10']} | {small['splade-whole']['recall@20']} | {small['splade-whole']['both-in-top2']} |",
        f"| SPLADE-v3 chunk-max | {small['splade-chunk-max']['recall@2']} | {small['splade-chunk-max']['recall@10']} | {small['splade-chunk-max']['recall@20']} | {small['splade-chunk-max']['both-in-top2']} |",
        f"| SPLADE++ EnsembleDistil whole | {pp_small.get('recall@2', '—')} | {pp_small.get('recall@10', '—')} | {pp_small.get('recall@20', '—')} | {pp_small.get('both-in-top2', '—')} |",
        "",
        "## limit (50k documents)",
        "",
        "| score | R@2 | R@10 | R@20 | R@100 | both@2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| BM25 | {full['bm25']['recall@2']} | {full['bm25']['recall@10']} | {full['bm25']['recall@20']} | {full['bm25']['recall@100']} | {full['bm25']['both-in-top2']} |",
        f"| SPLADE-v3 whole | {full['splade-whole']['recall@2']} | {full['splade-whole']['recall@10']} | {full['splade-whole']['recall@20']} | {full['splade-whole']['recall@100']} | {full['splade-whole']['both-in-top2']} |",
        f"| SPLADE++ EnsembleDistil whole | {pp_full.get('recall@2', '—')} | {pp_full.get('recall@10', '—')} | {pp_full.get('recall@20', '—')} | {pp_full.get('recall@100', '—')} | {pp_full.get('both-in-top2', '—')} |",
        f"| paper GTE-ModernColBERT | {paper['gte-moderncolbert-full']['recall@2']} | {paper['gte-moderncolbert-full']['recall@10']} | — | {paper['gte-moderncolbert-full']['recall@100']} | — |",
        "",
        "v3-DistilBERT MSMARCO MRR@10 is 38.7 vs BERT v3 40.2 vs ++ EnsembleDistil 38.0. "
        "The LIMIT comparison is against the same ++ checkpoint used earlier.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
