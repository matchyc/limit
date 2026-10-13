#!/usr/bin/env python3
"""Small LLM embedding model on official LIMIT queries.

Uses the same venv as MiniLM/SPLADE:
  experiments/zero-shot-embed/.venv/bin/python

Qwen3-Embedding-0.6B is a decoder-only embedding model (1024-d). Documents
are encoded once. Queries are encoded twice: the model's default retrieval
prompt, and a LIMIT-specific instruction.

Run:
  CUDA_VISIBLE_DEVICES=1 ../zero-shot-embed/.venv/bin/python run_llm_embed.py
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
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DOC_BATCH = 64
CASE_ATTRS = ("Art History", "Honey", "Psychology", "Go", "Birch Trees")
LIMIT_PROMPT = (
    "Instruct: Retrieve people whose likes list contains the exact attribute "
    "named in the query, not related topics or similar names.\nQuery:"
)

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


def load_split(path: Path):
    docs = load_jsonl(path / "corpus.jsonl")
    queries = load_jsonl(path / "queries.jsonl")
    qrels = load_jsonl(path / "qrels.jsonl")
    gold_map: dict[str, list[str]] = {}
    for row in qrels:
        gold_map.setdefault(row["query-id"], []).append(row["corpus-id"])
    doc_ids = [row["_id"] for row in docs]
    doc_texts = [doc_text(row) for row in docs]
    query_ids = [row["_id"] for row in queries]
    query_texts = [row["text"] for row in queries]
    id_to_pos = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    gold_index = np.asarray(
        [[id_to_pos[g] for g in gold_map[qid]] for qid in query_ids], dtype=np.int32
    )
    return doc_ids, doc_texts, query_ids, query_texts, gold_index, id_key(doc_ids)


def encode_texts(model, texts: list[str], batch: int, prompt_name=None, prompt=None) -> np.ndarray:
    kwargs = dict(
        batch_size=batch,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if prompt is not None:
        kwargs["prompt"] = prompt
    elif prompt_name is not None:
        kwargs["prompt_name"] = prompt_name
    chunks = []
    for start in range(0, len(texts), batch):
        end = min(start + batch, len(texts))
        chunks.append(model.encode(texts[start:end], **kwargs).astype(np.float32))
        if start == 0 or end == len(texts) or end % 3200 == 0:
            print(f"    encoded {end}/{len(texts)}", flush=True)
    return np.concatenate(chunks, axis=0)


def cosine_scores(q: np.ndarray, d: np.ndarray) -> np.ndarray:
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        qt = torch.from_numpy(q).to("cuda")
        dt = torch.from_numpy(d).to("cuda")
        scores = (qt @ dt.T).float().cpu().numpy()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return scores


def pack_cases(scores, query_texts, query_attrs, doc_ids, parsed, gold_index, key, k=8):
    cases = []
    want = {attr.lower(): i for i, attr in enumerate(query_attrs) if attr in CASE_ATTRS}
    for attr in CASE_ATTRS:
        row = want.get(attr.lower())
        if row is None:
            continue
        order = np.lexsort((key, -scores[row]))
        rank_of = {int(di): r + 1 for r, di in enumerate(order)}
        golds = set(int(g) for g in gold_index[row])
        top = []
        for rank, di in enumerate(order[:k], start=1):
            di = int(di)
            name, attrs = parsed[di]
            hits = [a for a in attrs if attr.lower() in a.lower() or a.lower() in attr.lower()]
            top.append({
                "rank": rank,
                "id": doc_ids[di],
                "score": round(float(scores[row, di]), 4),
                "is-gold": di in golds,
                "matching-likes-items": hits[:6],
            })
        cases.append({
            "query": query_texts[row],
            "attribute": attr,
            "gold-ranks": [rank_of[int(g)] for g in gold_index[row]],
            "top8": top,
        })
    return cases


def eval_split(name, path, model, do_bm25=True, cases=False):
    print(name, flush=True)
    doc_ids, doc_texts, query_ids, query_texts, gold_index, key = load_split(path)
    query_attrs = [attribute_of(t) for t in query_texts]
    print(f"  docs={len(doc_texts)} queries={len(query_texts)}", flush=True)
    sample = model.tokenizer(
        doc_texts[: min(200, len(doc_texts))],
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    n_tok = [len(ids) for ids in sample["input_ids"]]
    print(
        f"  sample tokens min={min(n_tok)} median={int(np.median(n_tok))} "
        f"max={max(n_tok)} model_max={model.max_seq_length}",
        flush=True,
    )

    out = {"docs": len(doc_texts), "queries": len(query_texts)}
    if do_bm25:
        bm25 = evaluate(
            BM25(doc_texts, doc_ids).score(query_texts).astype(np.float32),
            gold_index, key,
        )
        out["bm25"] = bm25
        print("  bm25", bm25, flush=True)

    print("  documents", flush=True)
    d_emb = encode_texts(model, doc_texts, DOC_BATCH)
    print("  queries default", flush=True)
    q_def = encode_texts(model, query_texts, 32, prompt_name="query")
    scores_def = cosine_scores(q_def, d_emb)
    out["llm-default-query-prompt"] = evaluate(scores_def, gold_index, key)
    print("  default", out["llm-default-query-prompt"], flush=True)

    print("  queries limit-instruction", flush=True)
    q_lim = encode_texts(model, query_texts, 32, prompt=LIMIT_PROMPT)
    scores_lim = cosine_scores(q_lim, d_emb)
    out["llm-limit-instruction"] = evaluate(scores_lim, gold_index, key)
    print("  limit-inst", out["llm-limit-instruction"], flush=True)

    if cases:
        parsed = [split_likes(t) for t in doc_texts]
        out["cases-default"] = pack_cases(
            scores_def, query_texts, query_attrs, doc_ids, parsed, gold_index, key,
        )
        out["cases-limit-instruction"] = pack_cases(
            scores_lim, query_texts, query_attrs, doc_ids, parsed, gold_index, key,
        )
        for case in out["cases-default"]:
            print(
                "  case", case["query"], "golds", case["gold-ranks"],
                "top1", case["top8"][0]["id"], "gold" if case["top8"][0]["is-gold"] else "fp",
                flush=True,
            )
    del d_emb, q_def, q_lim, scores_def, scores_lim
    torch.cuda.empty_cache()
    return out


def main() -> None:
    t0 = time.time()
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {MODEL_NAME} on {device}", flush=True)
    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
        processor_kwargs={"padding_side": "left"},
    )
    print(
        f"dim={model.get_embedding_dimension()} max_seq={model.max_seq_length} "
        f"prompts={model.prompts}",
        flush=True,
    )

    small = eval_split("limit-small", SMALL, model, do_bm25=True, cases=False)
    full = eval_split("limit-50k", FULL, model, do_bm25=True, cases=True)

    results = {
        "env": str(Path(sys.executable).resolve()),
        "model": MODEL_NAME,
        "dim": int(model.get_embedding_dimension()),
        "device": device,
        "seconds": round(time.time() - t0, 1),
        "limit-prompt": LIMIT_PROMPT,
        "limit-small": small,
        "limit": {k: v for k, v in full.items() if not k.startswith("cases")},
        "cases": {
            "default": full.get("cases-default"),
            "limit-instruction": full.get("cases-limit-instruction"),
        },
        "baselines": {
            "minilm-small-r@2": 16.2,
            "promptriever-small-r@2": 54.3,
            "splade++-small-r@2": 92.05,
            "bm25-50k-r@2": 92.65,
            "splade++-50k-r@2": 33.8,
            "splade-v3-distilbert-50k-r@2": 38.9,
            "promptriever-4096-50k-r@2": 3.0,
            "e5-mistral-50k-r@2": 1.3,
            "gritlm-50k-r@2": 2.4,
            "colbert-50k-r@2": 23.1,
        },
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    write_report(results, HERE / "report.md")
    print(f"wrote results in {results['seconds']}s", flush=True)


def write_report(res: dict, path: Path) -> None:
    small = res["limit-small"]
    full = res["limit"]
    b = res["baselines"]
    lines = [
        "# Qwen3-Embedding-0.6B on official LIMIT queries",
        "",
        f"Model `{res['model']}` ({res['dim']}-d) on {res['device']}. "
        f"Env `{res['env']}`. {res['seconds']}s.",
        "",
        "Queries use either the model's default retrieval prompt or a LIMIT-specific "
        "instruction. Documents are encoded with no prompt. Cosine similarity, "
        "official `Who likes X?` queries.",
        "",
        "## limit-small (46 documents)",
        "",
        "| score | R@2 | R@10 | R@20 | both@2 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| BM25 | {small['bm25']['recall@2']} | {small['bm25']['recall@10']} | {small['bm25']['recall@20']} | {small['bm25']['both-in-top2']} |",
        f"| Qwen3-Emb 0.6B default prompt | {small['llm-default-query-prompt']['recall@2']} | {small['llm-default-query-prompt']['recall@10']} | {small['llm-default-query-prompt']['recall@20']} | {small['llm-default-query-prompt']['both-in-top2']} |",
        f"| Qwen3-Emb 0.6B LIMIT instruction | {small['llm-limit-instruction']['recall@2']} | {small['llm-limit-instruction']['recall@10']} | {small['llm-limit-instruction']['recall@20']} | {small['llm-limit-instruction']['both-in-top2']} |",
        f"| MiniLM (prior) | {b['minilm-small-r@2']} | — | — | — |",
        f"| paper Promptriever 4096 | {b['promptriever-small-r@2']} | — | — | — |",
        f"| SPLADE++ whole (prior) | {b['splade++-small-r@2']} | — | — | — |",
        "",
        "## limit (50k documents)",
        "",
        "| score | R@2 | R@10 | R@20 | R@100 | both@2 | better gold median/mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| BM25 | {full['bm25']['recall@2']} | {full['bm25']['recall@10']} | {full['bm25']['recall@20']} | {full['bm25']['recall@100']} | {full['bm25']['both-in-top2']} | {full['bm25']['median-better-gold-rank']} / {full['bm25']['mean-better-gold-rank']} |",
        f"| Qwen3-Emb 0.6B default | {full['llm-default-query-prompt']['recall@2']} | {full['llm-default-query-prompt']['recall@10']} | {full['llm-default-query-prompt']['recall@20']} | {full['llm-default-query-prompt']['recall@100']} | {full['llm-default-query-prompt']['both-in-top2']} | {full['llm-default-query-prompt']['median-better-gold-rank']} / {full['llm-default-query-prompt']['mean-better-gold-rank']} |",
        f"| Qwen3-Emb 0.6B LIMIT inst. | {full['llm-limit-instruction']['recall@2']} | {full['llm-limit-instruction']['recall@10']} | {full['llm-limit-instruction']['recall@20']} | {full['llm-limit-instruction']['recall@100']} | {full['llm-limit-instruction']['both-in-top2']} | {full['llm-limit-instruction']['median-better-gold-rank']} / {full['llm-limit-instruction']['mean-better-gold-rank']} |",
        f"| SPLADE++ (prior) | {b['splade++-50k-r@2']} | — | — | — | — | — |",
        f"| SPLADE-v3 DistilBERT (prior) | {b['splade-v3-distilbert-50k-r@2']} | — | — | — | — | — |",
        f"| paper Promptriever 4096 | {b['promptriever-4096-50k-r@2']} | — | — | — | — | — |",
        f"| paper E5-Mistral 7B | {b['e5-mistral-50k-r@2']} | — | — | — | — | — |",
        f"| paper GTE-ModernColBERT | {b['colbert-50k-r@2']} | — | — | — | — | — |",
        "",
        "## Cases (50k, default prompt)",
        "",
    ]
    for case in res["cases"]["default"] or []:
        top1 = case["top8"][0]
        lines.append(
            f"- `{case['query']}` golds {case['gold-ranks']}; "
            f"top1 `{top1['id']}` gold={top1['is-gold']} hits={top1['matching-likes-items']}"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
