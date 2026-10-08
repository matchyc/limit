#!/usr/bin/env python3
"""Frozen MiniLM: multi-phrasing max, then divide by how common the hit is.

No new encoder. LIMIT-small only. Three measurements on one score:

1. Say each gloss (and, as a control, each original attribute) six ways.
   Chunk-max every phrasing and keep the best.
2. Divide that chunk score by how many documents sit in the same score band,
   and, separately, by how many documents have a chunk near the winning chunk.
   Plant 1, 2, or 5 exact-sentence copies and see whether the two golds stay up.
3. Do both at once, including gloss queries against those copies.

The score-band divisor is "how many documents match this query about as well."
The span divisor is "how many documents contain a chunk like this one."
Both readings are reported for every threshold. Recall on the original wording
is the control: it has to stay next to plain chunk-max.

Run:
  CUDA_VISIBLE_DEVICES=1 ../zero-shot-embed/.venv/bin/python run_span_specificity.py
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SMALL = ROOT / "data" / "limit-small"
SCHEME = ROOT / "experiments" / "scheme-failures"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

sys.path.insert(0, str(SCHEME))
from glosses import GLOSSES  # noqa: E402
from run_scheme_failures import (  # noqa: E402
    B,
    BM25,
    K1,
    attribute_of,
    doc_text,
    encode_sentences,
    gold_ranks,
    group_chunks,
    id_key,
    load_jsonl,
    lucene_idf,
    pack_metrics,
    render,
    split_likes,
    tokenize,
    validate_glosses,
)

TEMPLATES = (
    ("who-likes", "Who likes {x}?"),
    ("who-enjoys", "Who enjoys {x}?"),
    ("person-enjoys", "A person who enjoys {x}"),
    ("someone-likes", "Someone who likes {x}"),
    ("find-person", "Find a person who likes {x}"),
    ("which-person", "Which person likes {x}?"),
)
DELTAS = (0.0, 0.01, 0.02, 0.05, 0.10)
TAUS = (0.50, 0.70, 0.80, 0.90, 0.95)
COPY_N = 5


def pack(ranks: np.ndarray) -> dict:
    out = pack_metrics(ranks)
    out["both-in-top10"] = round(100.0 * float((ranks <= 10).all(axis=1).mean()), 2)
    return out


def ranks_of(scores: np.ndarray, gold_index: np.ndarray, key: np.ndarray) -> np.ndarray:
    return gold_ranks(scores, gold_index, key)


def evaluate(scores: np.ndarray, gold_index: np.ndarray, key: np.ndarray) -> dict:
    return pack(ranks_of(scores, gold_index, key))


def per_doc_max(token: np.ndarray, owner_cols: list[np.ndarray], n_docs: int
                ) -> tuple[np.ndarray, np.ndarray]:
    """Max over each document's chunks. Returns scores (Q, D) and winner chunk ids."""
    n_queries = token.shape[0]
    scores = np.full((n_queries, n_docs), -1e9, dtype=np.float32)
    winners = np.zeros((n_queries, n_docs), dtype=np.int32)
    rows = np.arange(n_queries)
    for doc_index, cols in enumerate(owner_cols):
        block = token[:, cols]
        local = block.argmax(axis=1)
        winners[:, doc_index] = cols[local]
        scores[:, doc_index] = block[rows, local]
    return scores, winners


def max_templates(template_tokens: list[np.ndarray], owner_cols: list[np.ndarray], n_docs: int
                  ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Max over phrasings and chunks. Also returns each phrasing's doc scores."""
    per = []
    best_scores = None
    best_winners = None
    for token in template_tokens:
        scores, winners = per_doc_max(token, owner_cols, n_docs)
        per.append(scores)
        if best_scores is None:
            best_scores = scores.copy()
            best_winners = winners.copy()
        else:
            better = scores > best_scores
            best_scores = np.maximum(best_scores, scores)
            best_winners = np.where(better, winners, best_winners)
    return best_scores, best_winners, per


def score_band_adjust(scores: np.ndarray, delta: float) -> np.ndarray:
    """Divide by how many documents have a score within ±delta, counting itself."""
    diff = np.abs(scores[:, :, None] - scores[:, None, :])
    df = (diff <= delta + 1e-6).sum(axis=-1).astype(np.float32)
    return scores / df


def chunk_doc_near(sim: np.ndarray, owner_cols: list[np.ndarray], tau: float) -> np.ndarray:
    """near[c, j] iff some chunk of document j has cosine ≥ tau with chunk c."""
    n_chunks = sim.shape[0]
    near = np.zeros((n_chunks, len(owner_cols)), dtype=bool)
    for doc_index, cols in enumerate(owner_cols):
        near[:, doc_index] = (sim[:, cols] >= tau).any(axis=1)
    return near


def span_df(near: np.ndarray, winners: np.ndarray) -> np.ndarray:
    return near[winners].sum(axis=-1).astype(np.float32)


def copy_to_docs(copy_to_chunk: np.ndarray, owner_cols: list[np.ndarray]) -> np.ndarray:
    """Max cosine from each planted copy to each original document's chunks."""
    n_queries, n_copies, _ = copy_to_chunk.shape
    out = np.empty((n_queries, n_copies, len(owner_cols)), dtype=np.float32)
    for doc_index, cols in enumerate(owner_cols):
        out[:, :, doc_index] = copy_to_chunk[:, :, cols].max(axis=-1)
    return out


def span_df_planted(near: np.ndarray, winners: np.ndarray, copy_to_chunk: np.ndarray,
                    copy_to_doc: np.ndarray, copy_copy: np.ndarray, tau: float, t: int
                    ) -> np.ndarray:
    """Document frequency of the winning span once t copies join the corpus."""
    n_queries, n_docs = winners.shape
    base = span_df(near, winners)
    gathered = np.empty((n_queries, n_docs, t), dtype=np.float32)
    for slot in range(t):
        gathered[:, :, slot] = np.take_along_axis(copy_to_chunk[:, slot, :], winners, axis=1)
    df_orig = base + (gathered >= tau).sum(axis=-1).astype(np.float32)
    near_orig = (copy_to_doc[:, :t, :] >= tau).sum(axis=-1).astype(np.float32)
    near_copies = (copy_copy[:, :t, :t] >= tau).sum(axis=-1).astype(np.float32)
    df_copy = near_orig + near_copies
    return np.concatenate([df_orig, df_copy], axis=1)


def percentile_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": round(float(values.mean()), 4),
        "median": round(float(np.median(values)), 4),
        "p10": round(float(np.quantile(values, 0.10)), 4),
        "p90": round(float(np.quantile(values, 0.90)), 4),
    }


def bm25_matrix(texts: list[str], queries: list[str]) -> np.ndarray:
    ids = [str(i) for i in range(len(texts))]
    return BM25(texts, ids).score(queries).astype(np.float32)


class FastBM25:
    """Same Lucene BM25 as the reference class, with the long documents tokenized once."""

    def __init__(self, texts: list[str]):
        self.base_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for doc_index, text in enumerate(texts):
            counts: dict[str, int] = {}
            tokens = tokenize(text)
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self.base_len.append(len(tokens))
            for token, tf in counts.items():
                self.postings.setdefault(token, []).append((doc_index, tf))
        self.n_base = len(texts)
        self._len_sum = float(sum(self.base_len))

    def score_one(self, query: str, extras: list[str]) -> np.ndarray:
        extra_len: list[int] = []
        extra_tf: list[dict[str, int]] = []
        for text in extras:
            counts: dict[str, int] = {}
            tokens = tokenize(text)
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            extra_len.append(len(tokens))
            extra_tf.append(counts)
        n_docs = self.n_base + len(extras)
        avgdl = (self._len_sum + sum(extra_len)) / n_docs
        scores = np.zeros(n_docs, dtype=np.float32)
        seen: set[str] = set()
        for token in tokenize(query):
            if token in seen:
                continue
            seen.add(token)
            posts = self.postings.get(token, [])
            hits = list(posts)
            for offset, counts in enumerate(extra_tf):
                tf = counts.get(token)
                if tf:
                    hits.append((self.n_base + offset, tf))
            if not hits:
                continue
            idf = lucene_idf(n_docs, len(hits))
            for doc_index, tf in hits:
                length = self.base_len[doc_index] if doc_index < self.n_base else extra_len[doc_index - self.n_base]
                norm = 1.0 - B + B * length / avgdl
                scores[doc_index] += idf * (tf * (K1 + 1.0)) / (tf + K1 * norm)
        return scores


def fmt_metrics(row: dict) -> str:
    return (
        f"R@2 {row['recall@2']} | R@10 {row['recall@10']} | R@20 {row['recall@20']} | "
        f"both@2 {row['both-in-top2']} | both@10 {row['both-in-top10']}"
    )


def check_score_band() -> None:
    scores = np.array([[0.90, 0.90, 0.50]], dtype=np.float32)
    adjusted = score_band_adjust(scores, 0.01)
    if not np.allclose(adjusted, [0.45, 0.45, 0.50]):
        raise SystemExit(f"score-band self-check failed: {adjusted}")


def main() -> None:
    t0 = time.time()
    check_score_band()
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
    for text, (name, attrs) in zip(doc_texts, parsed):
        if render(name, attrs) != text.strip():
            raise SystemExit(f"splitter did not round-trip {name}")

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
    small_key = id_key(doc_ids)

    gloss_index = np.asarray(
        [i for i, attr in enumerate(query_attrs) if attr in GLOSSES], dtype=np.int32
    )
    missing = sorted(set(GLOSSES) - set(query_attrs))
    if missing:
        raise SystemExit(f"glosses missing from queries: {missing[:5]}")
    n_docs = len(doc_ids)
    n_queries = len(query_ids)

    chunk_texts, owners = group_chunks(parsed, 1)
    owner_cols = [np.flatnonzero(owners == doc_index) for doc_index in range(n_docs)]
    print(
        f"docs={n_docs} queries={n_queries} chunks={len(chunk_texts)} glosses={len(gloss_index)}",
        flush=True,
    )

    import torch
    from sentence_transformers import SentenceTransformer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    model = SentenceTransformer(MODEL_NAME, device=device)

    template_query_texts = {
        name: [pattern.format(x=attr) for attr in query_attrs]
        for name, pattern in TEMPLATES
    }
    gloss_attrs = [query_attrs[i] for i in gloss_index.tolist()]
    template_gloss_texts = {
        name: [pattern.format(x=GLOSSES[attr]) for attr in gloss_attrs]
        for name, pattern in TEMPLATES
    }
    copy_ids = [f"Copy Person {i:02d}" for i in range(COPY_N)]
    copy_texts = [
        f"{copy_id} likes {attr}."
        for attr in query_attrs
        for copy_id in copy_ids
    ]

    encode_list = []
    spans: dict[str, tuple[int, int]] = {}

    def add_block(name: str, texts: list[str]) -> None:
        spans[name] = (len(encode_list), len(encode_list) + len(texts))
        encode_list.extend(texts)

    add_block("docs", doc_texts)
    add_block("chunks", chunk_texts)
    for name, texts in template_query_texts.items():
        add_block(f"q:{name}", texts)
    for name, texts in template_gloss_texts.items():
        add_block(f"g:{name}", texts)
    add_block("copies", copy_texts)
    print(f"encoding {len(encode_list)} strings", flush=True)
    vectors = encode_sentences(model, encode_list, batch_size=256)

    def take(name: str) -> np.ndarray:
        start, end = spans[name]
        return vectors[start:end]

    doc_emb = take("docs")
    chunk_emb = take("chunks")
    q_emb = {name: take(f"q:{name}") for name, _ in TEMPLATES}
    g_emb = {name: take(f"g:{name}") for name, _ in TEMPLATES}
    copy_emb = take("copies").reshape(n_queries, COPY_N, -1)

    def tokens_for(bank: dict[str, np.ndarray], side: np.ndarray) -> list[np.ndarray]:
        return [bank[name] @ side.T for name, _ in TEMPLATES]

    q_chunk_tokens = tokens_for(q_emb, chunk_emb)
    g_chunk_tokens = tokens_for(g_emb, chunk_emb)
    q_doc_tokens = tokens_for(q_emb, doc_emb)
    g_doc_tokens = tokens_for(g_emb, doc_emb)

    chunk_scores, chunk_winners, per_template_scores = max_templates(
        q_chunk_tokens, owner_cols, n_docs
    )
    gloss_scores, gloss_winners, gloss_per_template = max_templates(
        g_chunk_tokens, owner_cols, n_docs
    )
    whole_owners = [np.asarray([i], dtype=np.int32) for i in range(n_docs)]
    whole_scores, _, whole_per = max_templates(q_doc_tokens, whole_owners, n_docs)
    gloss_whole, _, gloss_whole_per = max_templates(g_doc_tokens, whole_owners, n_docs)
    # Whole-document "winners" above are dummy; the max over one column is the score.
    who = "who-likes"
    who_index = [name for name, _ in TEMPLATES].index(who)
    single_chunk = per_template_scores[who_index]
    single_winners = per_doc_max(q_chunk_tokens[who_index], owner_cols, n_docs)[1]
    single_whole = whole_per[who_index]
    gloss_single = gloss_per_template[who_index]
    gloss_single_winners = per_doc_max(g_chunk_tokens[who_index], owner_cols, n_docs)[1]
    gloss_single_whole = gloss_whole_per[who_index]

    print("bm25", flush=True)
    fast_bm25 = FastBM25(doc_texts)
    probe_scores = fast_bm25.score_one(query_texts[0], [copy_texts[0]])
    probe_ref = bm25_matrix(doc_texts + [copy_texts[0]], [query_texts[0]])[0]
    if not np.allclose(probe_scores, probe_ref, atol=1e-5):
        raise SystemExit(f"fast BM25 mismatch {probe_scores[:3]} vs {probe_ref[:3]}")

    bm25_per = {
        name: bm25_matrix(doc_texts, template_query_texts[name])
        for name, _ in TEMPLATES
    }
    bm25_single = bm25_per[who]
    bm25_max = np.maximum.reduce(list(bm25_per.values()))
    bm25_gloss_per = {
        name: bm25_matrix(doc_texts, template_gloss_texts[name])
        for name, _ in TEMPLATES
    }
    bm25_gloss_single = bm25_gloss_per[who]
    bm25_gloss_max = np.maximum.reduce(list(bm25_gloss_per.values()))

    gloss_gold = gold_index[gloss_index]
    gloss_key = small_key

    def sweep_band(scores: np.ndarray, golds: np.ndarray, key: np.ndarray) -> dict:
        return {f"{delta:.2f}": evaluate(score_band_adjust(scores, delta), golds, key)
                for delta in DELTAS}

    print("span neighborhoods", flush=True)
    sim = chunk_emb @ chunk_emb.T
    near_by_tau = {tau: chunk_doc_near(sim, owner_cols, tau) for tau in TAUS}

    def sweep_span(winners: np.ndarray, scores: np.ndarray, golds: np.ndarray, key: np.ndarray) -> dict:
        out = {}
        for tau, near in near_by_tau.items():
            df = span_df(near, winners)
            out[f"{tau:.2f}"] = evaluate(scores / df, golds, key)
        return out

    original = {
        "whole-doc": evaluate(single_whole, gold_index, small_key),
        "whole-doc-max-template": evaluate(whole_scores, gold_index, small_key),
        "bm25": evaluate(bm25_single, gold_index, small_key),
        "bm25-max-template": evaluate(bm25_max, gold_index, small_key),
        "chunk-max": evaluate(single_chunk, gold_index, small_key),
        "chunk-max-max-template": evaluate(chunk_scores, gold_index, small_key),
        "per-template-chunk": {
            name: evaluate(per_template_scores[i], gold_index, small_key)
            for i, (name, _) in enumerate(TEMPLATES)
        },
        "score-band": sweep_band(single_chunk, gold_index, small_key),
        "score-band-max-template": sweep_band(chunk_scores, gold_index, small_key),
        "span": sweep_span(single_winners, single_chunk, gold_index, small_key),
        "span-max-template": sweep_span(chunk_winners, chunk_scores, gold_index, small_key),
    }
    gloss = {
        "n": int(len(gloss_index)),
        "whole-doc": evaluate(gloss_single_whole, gloss_gold, gloss_key),
        "whole-doc-max-template": evaluate(gloss_whole, gloss_gold, gloss_key),
        "bm25": evaluate(bm25_gloss_single, gloss_gold, gloss_key),
        "bm25-max-template": evaluate(bm25_gloss_max, gloss_gold, gloss_key),
        "chunk-max": evaluate(gloss_single, gloss_gold, gloss_key),
        "chunk-max-max-template": evaluate(gloss_scores, gloss_gold, gloss_key),
        "per-template-chunk": {
            name: evaluate(gloss_per_template[i], gloss_gold, gloss_key)
            for i, (name, _) in enumerate(TEMPLATES)
        },
        "score-band": sweep_band(gloss_single, gloss_gold, gloss_key),
        "score-band-max-template": sweep_band(gloss_scores, gloss_gold, gloss_key),
        "span": sweep_span(gloss_single_winners, gloss_single, gloss_gold, gloss_key),
        "span-max-template": sweep_span(gloss_winners, gloss_scores, gloss_gold, gloss_key),
    }
    print("original chunk-max", original["chunk-max"], flush=True)
    print("gloss chunk-max", gloss["chunk-max"], "max-template", gloss["chunk-max-max-template"], flush=True)

    # Planted copies. copy_scores_*[:, k] is query · copy k, query-specific.
    q_who = q_emb[who]
    g_who = g_emb[who]
    copy_scores_who = np.einsum("qh,qth->qt", q_who, copy_emb)
    # Max over the six phrasings. Each phrasing of query q against copy (q, slot).
    copy_scores_max = np.empty_like(copy_scores_who)
    for slot in range(COPY_N):
        stacked = np.stack([
            (q_emb[name] * copy_emb[:, slot, :]).sum(axis=1) for name, _ in TEMPLATES
        ])
        copy_scores_max[:, slot] = stacked.max(axis=0)
    gloss_copy_emb = copy_emb[gloss_index]
    gloss_copy_who = np.einsum("gh,gth->gt", g_who, gloss_copy_emb)
    gloss_copy_max = np.empty_like(gloss_copy_who)
    for slot in range(COPY_N):
        stacked = np.stack([
            (g_emb[name] * gloss_copy_emb[:, slot, :]).sum(axis=1) for name, _ in TEMPLATES
        ])
        gloss_copy_max[:, slot] = stacked.max(axis=0)

    copy_to_chunk = np.einsum("qth,ch->qtc", copy_emb, chunk_emb)
    gloss_copy_to_chunk = copy_to_chunk[gloss_index]
    copy_copy = np.einsum("qth,qsh->qts", copy_emb, copy_emb)
    gloss_copy_copy = copy_copy[gloss_index]
    copy_doc = copy_to_docs(copy_to_chunk, owner_cols)
    gloss_copy_doc = copy_doc[gloss_index]

    # Which chunk did "Who likes Attr?" actually pick, and how close is a copy?
    gold_winner = single_winners[np.arange(n_queries)[:, None], gold_index]
    gold_vs_best_copy = np.empty(n_queries, dtype=np.float32)
    gold_pair = np.empty(n_queries, dtype=np.float32)
    for row in range(n_queries):
        pair = [copy_to_chunk[row, :, gold_winner[row, slot]].max() for slot in range(2)]
        gold_vs_best_copy[row] = max(pair)
        a, b = int(gold_winner[row, 0]), int(gold_winner[row, 1])
        gold_pair[row] = sim[a, b]
    copy_pair = copy_copy[:, 0, 1]
    score_gap = single_chunk[np.arange(n_queries)[:, None], gold_index].min(axis=1) - copy_scores_who.max(axis=1)

    def planted_block(base_scores: np.ndarray, winners: np.ndarray, copy_scores: np.ndarray,
                      whole: np.ndarray, whole_copy: np.ndarray,
                      queries_for_bm25: list[str], copy_text_rows: np.ndarray,
                      golds: np.ndarray, to_chunk: np.ndarray, to_doc: np.ndarray,
                      copies_sim: np.ndarray, label: str) -> dict:
        """copy_text_rows indexes the flat copy_texts list's query axis (global or gloss)."""
        print(f"planted {label}", flush=True)
        block = {}
        n_rows = base_scores.shape[0]
        flat_queries = queries_for_bm25
        # Map each row to COPY_N texts in copy_texts. copy_text_rows are global query indices.
        row_copy_texts = []
        for row in range(n_rows):
            src = int(copy_text_rows[row])
            row_copy_texts.append(copy_texts[src * COPY_N:(src + 1) * COPY_N])
        for t in (1, 2, 5):
            key = id_key(doc_ids + copy_ids[:t])
            dense = np.concatenate([base_scores, copy_scores[:, :t]], axis=1)
            whole_aug = np.concatenate([whole, whole_copy[:, :t]], axis=1)
            entry = {
                "chunk-max": evaluate(dense, golds, key),
                "whole-doc": evaluate(whole_aug, golds, key),
            }
            entry["score-band"] = {
                f"{delta:.2f}": evaluate(score_band_adjust(dense, delta), golds, key)
                for delta in DELTAS
            }
            entry["span"] = {}
            for tau, near in near_by_tau.items():
                df = span_df_planted(near, winners, to_chunk, to_doc, copies_sim, tau, t)
                entry["span"][f"{tau:.2f}"] = evaluate(dense / df, golds, key)
            print(f"  bm25 t={t}", flush=True)
            bm_scores = np.empty((n_rows, n_docs + t), dtype=np.float32)
            for row in range(n_rows):
                bm_scores[row] = fast_bm25.score_one(flat_queries[row], row_copy_texts[row][:t])
            entry["bm25"] = evaluate(bm_scores, golds, key)
            gaps = base_scores[np.arange(n_rows)[:, None], golds].min(axis=1) - copy_scores[:, :t].max(axis=1)
            entry["median-gap-worse-gold-minus-best-copy"] = round(float(np.median(gaps)), 4)
            entry["copies-above-worse-gold"] = int((gaps < 0).sum())
            same_band = {}
            for delta in DELTAS:
                same_band[f"{delta:.2f}"] = round(100.0 * float((np.abs(gaps) <= delta).mean()), 2)
            entry["queries-gold-and-best-copy-within-delta-pct"] = same_band
            block[f"t={t}"] = entry
            print(
                f"  t={t} chunk {entry['chunk-max']['recall@2']} "
                f"bm25 {entry['bm25']['recall@2']} whole {entry['whole-doc']['recall@2']}",
                flush=True,
            )
        return block

    planted = {
        "original-who-likes": planted_block(
            single_chunk, single_winners, copy_scores_who, single_whole, copy_scores_who,
            query_texts, np.arange(n_queries), gold_index,
            copy_to_chunk, copy_doc, copy_copy, "original",
        ),
        "original-max-template": planted_block(
            chunk_scores, chunk_winners, copy_scores_max, whole_scores, copy_scores_max,
            query_texts, np.arange(n_queries), gold_index,
            copy_to_chunk, copy_doc, copy_copy, "original-max",
        ),
        "gloss-who-likes": planted_block(
            gloss_single, gloss_single_winners, gloss_copy_who, gloss_single_whole, gloss_copy_who,
            template_gloss_texts[who], gloss_index, gloss_gold,
            gloss_copy_to_chunk, gloss_copy_doc, gloss_copy_copy, "gloss",
        ),
        "gloss-max-template": planted_block(
            gloss_scores, gloss_winners, gloss_copy_max, gloss_whole, gloss_copy_max,
            template_gloss_texts[who], gloss_index, gloss_gold,
            gloss_copy_to_chunk, gloss_copy_doc, gloss_copy_copy, "gloss-max",
        ),
    }

    # For the joint condition the BM25 baseline should see the same six phrasings.
    # Score each phrasing's augmented corpus and keep the max. Only t in {1,2,5}, gloss rows.
    print("bm25 max-template on planted glosses", flush=True)
    for t in (1, 2, 5):
        key = id_key(doc_ids + copy_ids[:t])
        acc = None
        for name, _ in TEMPLATES:
            texts = template_gloss_texts[name]
            bm_scores = np.empty((len(gloss_index), n_docs + t), dtype=np.float32)
            for row, src in enumerate(gloss_index.tolist()):
                extras = copy_texts[src * COPY_N:src * COPY_N + t]
                bm_scores[row] = fast_bm25.score_one(texts[row], extras)
            acc = bm_scores if acc is None else np.maximum(acc, bm_scores)
        planted["gloss-max-template"][f"t={t}"]["bm25-max-template"] = evaluate(acc, gloss_gold, key)
        # Original-word control uses templates that still contain the attribute.
        acc_o = None
        for name, _ in TEMPLATES:
            texts = template_query_texts[name]
            bm_scores = np.empty((n_queries, n_docs + t), dtype=np.float32)
            for row in range(n_queries):
                extras = copy_texts[row * COPY_N:row * COPY_N + t]
                bm_scores[row] = fast_bm25.score_one(texts[row], extras)
            acc_o = bm_scores if acc_o is None else np.maximum(acc_o, bm_scores)
        planted["original-max-template"][f"t={t}"]["bm25-max-template"] = evaluate(acc_o, gold_index, key)
        print(f"  t={t} gloss bm25-max {planted['gloss-max-template'][f't={t}']['bm25-max-template']['recall@2']}", flush=True)

    off = sim[np.triu_indices(sim.shape[0], k=1)]
    rng = np.random.default_rng(0)
    sample = off if off.size <= 200000 else off[rng.choice(off.size, 200000, replace=False)]
    winner_has_attr = 0
    for row, attr in enumerate(query_attrs):
        top = int(np.lexsort((small_key, -single_chunk[row]))[0])
        if attr in chunk_texts[int(single_winners[row, top])]:
            winner_has_attr += 1

    results = {
        "model": MODEL_NAME,
        "device": device,
        "docs": n_docs,
        "queries": n_queries,
        "chunks": len(chunk_texts),
        "glosses": int(len(gloss_index)),
        "templates": [{"name": name, "pattern": pattern} for name, pattern in TEMPLATES],
        "deltas": list(DELTAS),
        "taus": list(TAUS),
        "seconds": round(time.time() - t0, 1),
        "original": original,
        "gloss": gloss,
        "planted": planted,
        "diagnostics": {
            "chunk-cosine-offdiag-sample": percentile_stats(sample),
            "gold-winning-chunk-vs-best-copy": percentile_stats(gold_vs_best_copy),
            "two-golds-winning-chunks": percentile_stats(gold_pair),
            "two-copies": percentile_stats(copy_pair),
            "score-gap-worse-gold-minus-best-copy": percentile_stats(score_gap),
            "top1-chunk-contains-attribute": {
                "hits": winner_has_attr,
                "queries": n_queries,
            },
        },
    }
    out_json = HERE / "results.json"
    out_json.write_text(json.dumps(results, indent=2) + "\n")
    write_report(results, HERE / "report.md")
    print(f"wrote {out_json} in {results['seconds']}s", flush=True)


def write_report(res: dict, path: Path) -> None:
    lines = [
        "# Span specificity on frozen MiniLM",
        "",
        f"Model `{res['model']}` on {res['device']}. "
        f"LIMIT-small: {res['docs']} docs, {res['queries']} queries, {res['chunks']} attribute chunks, "
        f"{res['glosses']} gloss queries. {res['seconds']}s.",
        "",
        "One score. Chunk-max over a phrasing, optionally the max over six phrasings, "
        "then divide by a document frequency. "
        "Score-band df counts documents whose chunk-max lies within ±δ (including itself). "
        "Span df counts documents that have a chunk with cosine ≥ τ to the winning chunk.",
        "",
        "Recall is macro over the two golds, in percent.",
        "",
        "## Original wording (control)",
        "",
        "| score | R@2 | R@10 | R@20 | both@2 | both@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in (
        "whole-doc", "whole-doc-max-template", "bm25", "bm25-max-template",
        "chunk-max", "chunk-max-max-template",
    ):
        row = res["original"][label]
        lines.append(
            f"| {label} | {row['recall@2']} | {row['recall@10']} | {row['recall@20']} | "
            f"{row['both-in-top2']} | {row['both-in-top10']} |"
        )
    lines += ["", "### Score-band divisor on chunk-max", "", "| δ | R@2 | R@10 | both@2 |", "| ---: | ---: | ---: | ---: |"]
    for key, row in res["original"]["score-band"].items():
        lines.append(f"| {key} | {row['recall@2']} | {row['recall@10']} | {row['both-in-top2']} |")
    lines += ["", "### Span divisor on chunk-max", "", "| τ | R@2 | R@10 | both@2 |", "| ---: | ---: | ---: | ---: |"]
    for key, row in res["original"]["span"].items():
        lines.append(f"| {key} | {row['recall@2']} | {row['recall@10']} | {row['both-in-top2']} |")

    lines += ["", "## Gloss queries", "", "| score | R@2 | R@10 | R@20 | both@2 | both@10 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label in (
        "whole-doc", "whole-doc-max-template", "bm25", "bm25-max-template",
        "chunk-max", "chunk-max-max-template",
    ):
        row = res["gloss"][label]
        lines.append(
            f"| {label} | {row['recall@2']} | {row['recall@10']} | {row['recall@20']} | "
            f"{row['both-in-top2']} | {row['both-in-top10']} |"
        )
    lines += ["", "Per phrasing, chunk-max:", ""]
    for name, row in res["gloss"]["per-template-chunk"].items():
        lines.append(f"- {name}: {fmt_metrics(row)}")
    lines += ["", "### Score-band on gloss chunk-max", "", "| δ | single R@2 | single R@10 | max-template R@2 | max-template R@10 |", "| ---: | ---: | ---: | ---: | ---: |"]
    for key in res["gloss"]["score-band"]:
        a = res["gloss"]["score-band"][key]
        b = res["gloss"]["score-band-max-template"][key]
        lines.append(f"| {key} | {a['recall@2']} | {a['recall@10']} | {b['recall@2']} | {b['recall@10']} |")
    lines += ["", "### Span divisor on gloss chunk-max", "", "| τ | single R@2 | single R@10 | max-template R@2 | max-template R@10 |", "| ---: | ---: | ---: | ---: | ---: |"]
    for key in res["gloss"]["span"]:
        a = res["gloss"]["span"][key]
        b = res["gloss"]["span-max-template"][key]
        lines.append(f"| {key} | {a['recall@2']} | {a['recall@10']} | {b['recall@2']} | {b['recall@10']} |")

    lines += ["", "## Planted exact copies", ""]
    lines.append(
        "Each query is scored against the 46 documents plus t other people whose only sentence is "
        "`Copy Person i likes {attribute}.` Gap is the worse gold's chunk score minus the best copy."
    )
    for family in ("original-who-likes", "original-max-template", "gloss-who-likes", "gloss-max-template"):
        lines += ["", f"### {family}", ""]
        lines.append("| t | chunk R@2 | chunk R@10 | whole R@10 | bm25 R@2 | bm25 R@10 | median gap |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for t_name, entry in res["planted"][family].items():
            lines.append(
                f"| {t_name} | {entry['chunk-max']['recall@2']} | {entry['chunk-max']['recall@10']} | "
                f"{entry['whole-doc']['recall@10']} | {entry['bm25']['recall@2']} | {entry['bm25']['recall@10']} | "
                f"{entry['median-gap-worse-gold-minus-best-copy']} |"
            )
        lines += [
            "",
            "With t=5 the corpus has 51 documents, so R@10 still counts a gold that sits just behind the copies. "
            "R@2 is the cut that shows whether those copies took the top.",
            "",
            "Score-band R@2:",
            "",
        ]
        header = "| δ | " + " | ".join(res["planted"][family]) + " |"
        lines.append(header)
        lines.append("| ---: | " + " | ".join("---:" for _ in res["planted"][family]) + " |")
        for delta in res["deltas"]:
            cells = [res["planted"][family][t_name]["score-band"][f"{delta:.2f}"]["recall@2"]
                     for t_name in res["planted"][family]]
            lines.append("| " + f"{delta:.2f}" + " | " + " | ".join(str(c) for c in cells) + " |")
        lines += ["", "Span R@2:", ""]
        lines.append(header.replace("δ", "τ"))
        lines.append("| ---: | " + " | ".join("---:" for _ in res["planted"][family]) + " |")
        for tau in res["taus"]:
            cells = [res["planted"][family][t_name]["span"][f"{tau:.2f}"]["recall@2"]
                     for t_name in res["planted"][family]]
            lines.append("| " + f"{tau:.2f}" + " | " + " | ".join(str(c) for c in cells) + " |")
        if "bm25-max-template" in next(iter(res["planted"][family].values())):
            bits = [
                f"{t_name} R@10 {res['planted'][family][t_name]['bm25-max-template']['recall@10']}"
                for t_name in res["planted"][family]
            ]
            lines += ["", "BM25 given the same six phrasings, then max: " + "; ".join(bits), ""]

    diag = res["diagnostics"]
    lines += ["", "## Diagnostics", ""]
    for label, row in diag.items():
        if isinstance(row, dict) and "median" in row:
            lines.append(
                f"- {label}: n={row['n']} mean {row['mean']} median {row['median']} "
                f"p10 {row['p10']} p90 {row['p90']}"
            )
        else:
            lines.append(f"- {label}: {row}")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
