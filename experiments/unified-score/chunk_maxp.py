"""Chunk-MaxP vs whole-doc mean pooling on LIMIT-small.

Same encoder (all-MiniLM-L6-v2), only the aggregation changes:
  baseline: cos(query, full document vector)
  chunk-max: max over attribute-span chunks cos(query, chunk vector)

A chunk is "<Name> likes <attr>." so each of the ~50 attributes keeps its own
vector. Run with the cached model, e.g.:
  CUDA_VISIBLE_DEVICES=1 ../zero-shot-embed/.venv/bin/python chunk_maxp.py
Writes results.json and report.md next to this script.
"""
import json
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SMALL = ROOT / "data" / "limit-small"


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_likes(text):
    """'Name likes A, B and C.' -> (name, [A, B, C])."""
    m = re.match(r"^(.*?) likes (.*)\.\s*$", text)
    if not m:
        return text, [text]
    name, rest = m.group(1), m.group(2)
    parts = re.split(r",\s*|\s+and\s+", rest)
    attrs = [p.strip() for p in parts if p.strip()]
    return name, attrs


def recall_at(ranks, gold, k):
    hits = sum(1 for g in gold if ranks.get(g, 10**9) < k)
    return hits / len(gold)


def main():
    import torch
    from sentence_transformers import SentenceTransformer

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    queries = read_jsonl(SMALL / "queries.jsonl")
    corpus = read_jsonl(SMALL / "corpus.jsonl")
    qrels = read_jsonl(SMALL / "qrels.jsonl")
    gold = {}
    for r in qrels:
        gold.setdefault(r["query-id"], []).append(r["corpus-id"])

    doc_text = {d["_id"]: d["text"] for d in corpus}
    doc_ids = [d["_id"] for d in corpus]

    # Build chunks: one sentence per attribute, keeping the name for context.
    chunk_texts, chunk_owner = [], []
    for did in doc_ids:
        name, attrs = split_likes(doc_text[did])
        for a in attrs:
            chunk_texts.append(f"{name} likes {a}.")
            chunk_owner.append(did)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    q_emb = model.encode([q["text"] for q in queries], normalize_embeddings=True,
                         show_progress_bar=False, convert_to_numpy=True)
    d_emb = model.encode([doc_text[i] for i in doc_ids], normalize_embeddings=True,
                         show_progress_bar=False, convert_to_numpy=True,
                         batch_size=64)
    c_emb = model.encode(chunk_texts, normalize_embeddings=True,
                         show_progress_bar=False, convert_to_numpy=True,
                         batch_size=256)

    import numpy as np
    q_emb, d_emb, c_emb = map(np.asarray, (q_emb, d_emb, c_emb))
    owner_idx = np.array([doc_ids.index(o) for o in chunk_owner])

    res = {"model": "sentence-transformers/all-MiniLM-L6-v2", "device": device,
           "n_docs": len(doc_ids), "n_queries": len(queries),
           "n_chunks": len(chunk_texts)}
    for label, sims in (("whole-doc", q_emb @ d_emb.T),
                        ("chunk-max", None)):
        if sims is None:
            tok = q_emb @ c_emb.T  # (Q, C)
            sims = np.full((len(queries), len(doc_ids)), -1e9)
            # max over chunks per doc: loop docs (46) to keep memory flat
            for j in range(len(doc_ids)):
                m = owner_idx == j
                if m.any():
                    sims[:, j] = tok[:, m].max(axis=1)
        r2 = r10 = r20 = 0.0
        both_top2 = 0
        for i, q in enumerate(queries):
            order = np.argsort(-sims[i], kind="stable")
            rank = {doc_ids[j]: r for r, j in enumerate(order)}
            g = gold[q["_id"]]
            r2 += recall_at(rank, g, 2)
            r10 += recall_at(rank, g, 10)
            r20 += recall_at(rank, g, 20)
            if all(rank[x] < 2 for x in g):
                both_top2 += 1
        n = len(queries)
        res[label] = {"recall@2": round(100 * r2 / n, 2),
                      "recall@10": round(100 * r10 / n, 2),
                      "recall@20": round(100 * r20 / n, 2),
                      "both-in-top2": round(100 * both_top2 / n, 2)}
    res["seconds"] = round(time.time() - t0, 1)

    (HERE / "results.json").write_text(json.dumps(res, indent=2))
    lines = ["# Chunk-MaxP vs whole-doc (LIMIT-small, MiniLM-L6-v2)", "",
             f"chunks: {res['n_chunks']} over {res['n_docs']} docs", "",
             "| scoring | R@2 | R@10 | R@20 | both-in-top2 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for label in ("whole-doc", "chunk-max"):
        m = res[label]
        lines.append(f"| {label} | {m['recall@2']} | {m['recall@10']} | "
                     f"{m['recall@20']} | {m['both-in-top2']} |")
    lines += ["", f"device={device}, seconds={res['seconds']}"]
    (HERE / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
