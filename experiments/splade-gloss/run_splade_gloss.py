#!/usr/bin/env python3
"""SPLADE++ (naver/splade-cocondenser-ensembledistil) on LIMIT-small.

Problem 2: the query is a definition with no shared content token, the document
still names the attribute. BM25 is 3.57 R@2 on that slice. This run asks whether
MLM expansion puts the attribute token (or the definition) into the sparse
vector so the gold documents rank in the top 2.

Whole-document max-pool is the production sparse score. Oracle per-attribute
chunks are a control: if they match whole-document, pooling is not the failure.

Run:
  CUDA_VISIBLE_DEVICES=1 ../zero-shot-embed/.venv/bin/python run_splade_gloss.py
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
SCHEME = ROOT / "experiments" / "scheme-failures"
MODEL_NAME = "naver/splade-cocondenser-ensembledistil"

sys.path.insert(0, str(SCHEME))
from glosses import GLOSSES  # noqa: E402
from run_scheme_failures import (  # noqa: E402
    BM25,
    attribute_of,
    content_tokens,
    doc_text,
    gold_ranks,
    group_chunks,
    id_key,
    load_jsonl,
    pack_metrics,
    render,
    split_likes,
    validate_glosses,
)

EXAMPLE_ATTRS = (
    "Ham", "Joshua Trees", "Havarti", "Chairs", "Ancient Egypt",
    "Limes", "Disco Music", "Soy Sauce", "Barley", "Honey",
)


def pack(ranks: np.ndarray) -> dict:
    out = pack_metrics(ranks)
    out["both-in-top10"] = round(100.0 * float((ranks <= 10).all(axis=1).mean()), 2)
    return out


def evaluate(scores: np.ndarray, gold_index: np.ndarray, key: np.ndarray) -> dict:
    return pack(gold_ranks(scores, gold_index, key))


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "to_dense"):
        value = value.to_dense()
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def encode_sparse(model, texts: list[str], batch_size: int = 32):
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        convert_to_sparse_tensor=True,
        show_progress_bar=False,
    )


def score_matrix(model, queries, docs) -> np.ndarray:
    sim = model.similarity(queries, docs)
    return np.asarray(to_numpy(sim), dtype=np.float32)


def max_over_chunks(q_emb, c_emb, owners: np.ndarray, n_docs: int, model) -> np.ndarray:
    token = score_matrix(model, q_emb, c_emb)
    scores = np.full((token.shape[0], n_docs), -1e9, dtype=np.float32)
    for doc_index in range(n_docs):
        cols = owners == doc_index
        if not np.any(cols):
            continue
        scores[:, doc_index] = token[:, cols].max(axis=1)
    return scores


def term_map(decoded_row: list[tuple[str, float]]) -> dict[str, float]:
    return {token.lower(): float(weight) for token, weight in decoded_row}


def attribute_in_expansion(attr: str, terms: dict[str, float]) -> dict:
    tokens = sorted(content_tokens(attr))
    hits = [tok for tok in tokens if tok in terms]
    return {
        "attribute-content-tokens": tokens,
        "tokens-in-expansion": hits,
        "any": bool(hits),
        "weights": {tok: round(terms[tok], 4) for tok in hits},
    }


def main() -> None:
    t0 = time.time()
    bad = validate_glosses(GLOSSES)
    if bad:
        raise SystemExit(f"glosses share tokens with attributes: {bad[:3]}")

    docs = load_jsonl(SMALL / "corpus.jsonl")
    queries = load_jsonl(SMALL / "queries.jsonl")
    qrels = load_jsonl(SMALL / "qrels.jsonl")
    gold_map: dict[str, list[str]] = {}
    for row in qrels:
        gold_map.setdefault(row["query-id"], []).append(row["corpus-id"])

    doc_ids = [row["_id"] for row in docs]
    doc_texts = [doc_text(row) for row in docs]
    parsed = [split_likes(text) for text in doc_texts]
    query_texts = [row["text"] for row in queries]
    query_ids = [row["_id"] for row in queries]
    query_attrs = [attribute_of(text) for text in query_texts]
    id_to_pos = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    gold_index = np.asarray(
        [[id_to_pos[g] for g in gold_map[qid]] for qid in query_ids], dtype=np.int32
    )
    key = id_key(doc_ids)
    n_docs = len(doc_ids)

    gloss_index = np.asarray(
        [i for i, attr in enumerate(query_attrs) if attr in GLOSSES], dtype=np.int32
    )
    missing = sorted(set(GLOSSES) - set(query_attrs))
    if missing:
        raise SystemExit(f"glosses missing: {missing[:5]}")
    gloss_gold = gold_index[gloss_index]
    gloss_attrs = [query_attrs[i] for i in gloss_index.tolist()]
    gloss_queries = [f"Who likes {GLOSSES[attr]}?" for attr in gloss_attrs]
    gloss_only = [GLOSSES[attr] for attr in gloss_attrs]
    rewritten = [
        render(name, [GLOSSES.get(attr, attr) for attr in attrs])
        for name, attrs in parsed
    ]
    chunk_texts, owners = group_chunks(parsed, 1)

    print(
        f"docs={n_docs} queries={len(query_texts)} glosses={len(gloss_index)} "
        f"chunks={len(chunk_texts)}",
        flush=True,
    )

    bm25 = BM25(doc_texts, doc_ids)
    bm25_rewritten = BM25(rewritten, doc_ids)
    bm25_original = evaluate(bm25.score(query_texts).astype(np.float32), gold_index, key)
    bm25_gloss = evaluate(bm25.score(gloss_queries).astype(np.float32), gloss_gold, key)
    bm25_gloss_only = evaluate(bm25.score(gloss_only).astype(np.float32), gloss_gold, key)
    bm25_orig_on_rewritten = evaluate(
        bm25_rewritten.score(query_texts).astype(np.float32), gold_index, key
    )
    print("bm25 original", bm25_original, "gloss", bm25_gloss, flush=True)

    from sentence_transformers import SparseEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {MODEL_NAME} on {device}", flush=True)
    model = SparseEncoder(MODEL_NAME, device=device)
    tokenizer = model.tokenizer
    doc_tok = tokenizer(doc_texts, add_special_tokens=True, truncation=False)
    n_tokens = [len(ids) for ids in doc_tok["input_ids"]]
    truncated = sum(n > tokenizer.model_max_length for n in n_tokens)
    print(
        f"doc tokens min={min(n_tokens)} median={int(np.median(n_tokens))} "
        f"max={max(n_tokens)} truncated_at_{tokenizer.model_max_length}={truncated}",
        flush=True,
    )

    print("encoding", flush=True)
    d_whole = encode_sparse(model, doc_texts, batch_size=8)
    d_rewritten = encode_sparse(model, rewritten, batch_size=8)
    d_chunks = encode_sparse(model, chunk_texts, batch_size=32)
    q_orig = encode_sparse(model, query_texts, batch_size=32)
    q_gloss = encode_sparse(model, gloss_queries, batch_size=32)
    q_gloss_only = encode_sparse(model, gloss_only, batch_size=32)

    orig_whole = score_matrix(model, q_orig, d_whole)
    orig_chunk = max_over_chunks(q_orig, d_chunks, owners, n_docs, model)
    gloss_whole = score_matrix(model, q_gloss, d_whole)
    gloss_chunk = max_over_chunks(q_gloss, d_chunks, owners, n_docs, model)
    gloss_only_whole = score_matrix(model, q_gloss_only, d_whole)
    orig_on_rewritten = score_matrix(model, q_orig, d_rewritten)

    original = {
        "bm25": bm25_original,
        "splade-whole": evaluate(orig_whole, gold_index, key),
        "splade-chunk-max": evaluate(orig_chunk, gold_index, key),
    }
    # Reverse of problem 2, only on the 126 glossed attributes: query still
    # says Ham, documents now say the definition.
    orig_q_rewritten_docs = {
        "bm25": evaluate(
            bm25_rewritten.score([query_texts[i] for i in gloss_index.tolist()]).astype(np.float32),
            gloss_gold, key,
        ),
        "splade-whole": evaluate(orig_on_rewritten[gloss_index], gloss_gold, key),
        "bm25-on-original-docs": evaluate(
            bm25.score([query_texts[i] for i in gloss_index.tolist()]).astype(np.float32),
            gloss_gold, key,
        ),
        "splade-on-original-docs": evaluate(orig_whole[gloss_index], gloss_gold, key),
    }
    gloss = {
        "n": int(len(gloss_index)),
        "bm25": bm25_gloss,
        "bm25-gloss-only": bm25_gloss_only,
        "splade-whole": evaluate(gloss_whole, gloss_gold, key),
        "splade-chunk-max": evaluate(gloss_chunk, gloss_gold, key),
        "splade-gloss-only": evaluate(gloss_only_whole, gloss_gold, key),
    }
    print("original", original["splade-whole"], flush=True)
    print("gloss", gloss["splade-whole"], flush=True)

    decoded_gloss = model.decode(q_gloss, top_k=40)
    decoded_only = model.decode(q_gloss_only, top_k=40)
    decoded_docs = model.decode(d_whole, top_k=80)
    gold0 = [int(gloss_gold[i, 0]) for i in range(len(gloss_index))]
    decoded_gold = [decoded_docs[i] for i in gold0]

    expansions = []
    attr_in_q = []
    attr_in_q_only = []
    attr_in_gold = []
    for i, attr in enumerate(gloss_attrs):
        q_terms = term_map(decoded_gloss[i])
        only_terms = term_map(decoded_only[i])
        gold_terms = term_map(decoded_gold[i])
        q_hit = attribute_in_expansion(attr, q_terms)
        only_hit = attribute_in_expansion(attr, only_terms)
        gold_hit = attribute_in_expansion(attr, gold_terms)
        attr_in_q.append(q_hit["any"])
        attr_in_q_only.append(only_hit["any"])
        attr_in_gold.append(gold_hit["any"])
        expansions.append({
            "attribute": attr,
            "gloss": GLOSSES[attr],
            "query-expansion-has-attribute": q_hit,
            "gloss-only-expansion-has-attribute": only_hit,
            "gold-doc-expansion-has-attribute": gold_hit,
            "query-top-terms": [
                [tok, round(w, 3)] for tok, w in decoded_gloss[i][:12]
            ],
        })

    gloss["query-expansion-contains-attribute"] = {
        "hits": int(sum(attr_in_q)),
        "n": int(len(gloss_attrs)),
        "rate": round(100.0 * float(np.mean(attr_in_q)), 2),
    }
    gloss["gloss-only-expansion-contains-attribute"] = {
        "hits": int(sum(attr_in_q_only)),
        "n": int(len(gloss_attrs)),
        "rate": round(100.0 * float(np.mean(attr_in_q_only)), 2),
    }
    gloss["gold-doc-expansion-contains-attribute"] = {
        "hits": int(sum(attr_in_gold)),
        "n": int(len(gloss_attrs)),
        "rate": round(100.0 * float(np.mean(attr_in_gold)), 2),
    }

    # Recall split: queries whose expansion recovered a content token of the attribute.
    hit_rows = np.asarray([i for i, ok in enumerate(attr_in_q) if ok], dtype=np.int32)
    miss_rows = np.asarray([i for i, ok in enumerate(attr_in_q) if not ok], dtype=np.int32)
    if len(hit_rows):
        gloss["splade-whole-when-query-has-attribute"] = evaluate(
            gloss_whole[hit_rows], gloss_gold[hit_rows], key
        )
    if len(miss_rows):
        gloss["splade-whole-when-query-lacks-attribute"] = evaluate(
            gloss_whole[miss_rows], gloss_gold[miss_rows], key
        )

    examples = []
    attr_to_row = {attr: i for i, attr in enumerate(gloss_attrs)}
    for attr in EXAMPLE_ATTRS:
        if attr not in attr_to_row:
            continue
        i = attr_to_row[attr]
        ranks = gold_ranks(gloss_whole[i : i + 1], gloss_gold[i : i + 1], key)[0]
        examples.append({
            "attribute": attr,
            "gloss": GLOSSES[attr],
            "gold-ranks": ranks.tolist(),
            "query-has-attribute-token": expansions[i]["query-expansion-has-attribute"],
            "query-top-terms": expansions[i]["query-top-terms"],
        })

    results = {
        "model": MODEL_NAME,
        "device": device,
        "docs": n_docs,
        "queries": len(query_texts),
        "glosses": int(len(gloss_index)),
        "doc-token-length": {
            "min": int(min(n_tokens)),
            "median": int(np.median(n_tokens)),
            "max": int(max(n_tokens)),
            "model-max-length": int(tokenizer.model_max_length),
            "truncated": int(truncated),
        },
        "seconds": round(time.time() - t0, 1),
        "original": original,
        "original-query-on-glossed-docs": orig_q_rewritten_docs,
        "gloss": gloss,
        "examples": examples,
        "expansions": expansions,
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    write_report(results, HERE / "report.md")
    print(f"wrote results in {results['seconds']}s", flush=True)


def write_report(res: dict, path: Path) -> None:
    g = res["gloss"]
    o = res["original"]
    lines = [
        "# SPLADE++ on LIMIT-small original and gloss queries",
        "",
        f"Model `{res['model']}` on {res['device']}. "
        f"{res['docs']} docs, {res['queries']} original queries, {res['glosses']} gloss queries. "
        f"{res['seconds']}s. Whole documents are {res['doc-token-length']['median']} tokens "
        f"median, max {res['doc-token-length']['max']}; none truncated at "
        f"{res['doc-token-length']['model-max-length']}.",
        "",
        "Recall is macro over the two golds, in percent.",
        "",
        "## Original wording",
        "",
        "| score | R@2 | R@10 | both@2 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label in ("bm25", "splade-whole", "splade-chunk-max"):
        row = o[label]
        lines.append(
            f"| {label} | {row['recall@2']} | {row['recall@10']} | {row['both-in-top2']} |"
        )
    rev = res["original-query-on-glossed-docs"]
    lines += [
        "",
        "The 126 attributes that have glosses, original query, documents rewritten to the definition:",
        "",
        "| score | R@2 | R@10 | both@2 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label in (
        "bm25-on-original-docs", "splade-on-original-docs",
        "bm25", "splade-whole",
    ):
        row = rev[label]
        lines.append(
            f"| {label} | {row['recall@2']} | {row['recall@10']} | {row['both-in-top2']} |"
        )
    lines += [
        "",
        "## Gloss queries (`Who likes {definition}?`)",
        "",
        "| score | R@2 | R@10 | both@2 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label in (
        "bm25", "bm25-gloss-only", "splade-whole", "splade-chunk-max", "splade-gloss-only",
    ):
        row = g[label]
        lines.append(
            f"| {label} | {row['recall@2']} | {row['recall@10']} | {row['both-in-top2']} |"
        )
    lines += [
        "",
        "Query expansion contains a content token of the original attribute: "
        f"{g['query-expansion-contains-attribute']['hits']}/{g['query-expansion-contains-attribute']['n']} "
        f"({g['query-expansion-contains-attribute']['rate']}%). "
        "Gloss text alone: "
        f"{g['gloss-only-expansion-contains-attribute']['rate']}%. "
        "First gold document still has the attribute token: "
        f"{g['gold-doc-expansion-contains-attribute']['rate']}%.",
        "",
    ]
    if "splade-whole-when-query-has-attribute" in g:
        row = g["splade-whole-when-query-has-attribute"]
        lines.append(
            f"When the query expansion recovered the attribute, SPLADE whole R@2 "
            f"{row['recall@2']}, R@10 {row['recall@10']}."
        )
    if "splade-whole-when-query-lacks-attribute" in g:
        row = g["splade-whole-when-query-lacks-attribute"]
        lines.append(
            f"When it did not, SPLADE whole R@2 {row['recall@2']}, R@10 {row['recall@10']}."
        )
    lines += ["", "## Examples", ""]
    for ex in res["examples"]:
        lines.append(
            f"- **{ex['attribute']}** ← {ex['gloss']}. gold ranks {ex['gold-ranks']}. "
            f"attribute in query expansion: {ex['query-has-attribute-token']['any']} "
            f"{ex['query-has-attribute-token']['weights']}. "
            f"top terms: {ex['query-top-terms'][:8]}"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
