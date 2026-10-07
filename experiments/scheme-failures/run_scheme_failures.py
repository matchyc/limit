#!/usr/bin/env python3
"""Failure modes of existing retrievers on LIMIT-small.

Same encoder everywhere: sentence-transformers/all-MiniLM-L6-v2.
LIMIT-small (46 docs) and data/limit (50k) are different draws; distractors
are a seed-0 permutation of the 50k file, matching experiments/zero-shot-embed.

Claims this script separates, without training a new model:

1. Oracle per-attribute chunks ("Name likes Attr.") make chunk-MaxP look solved.
   Break that by paraphrases that drop the rare token, by coarser / misaligned
   chunks, and by attributes that are string-prefixes of other likes-items.
2. Late fusion cannot resurrect a gold missed by both a lexical channel and a
   whole-document dense channel. RRF and union recall are computed on the
   existing scores.
3. A crude late-interaction proxy: MiniLM token vectors, summed MaxSim, plus
   late chunking (mean-pool contextual token vectors inside each attribute
   span, then max over spans).
4. Chunk-MaxP still drowns once distractor documents from the other draw
   contain the attribute string, or once exact-string copies are planted.

Run:
  CUDA_VISIBLE_DEVICES=1 ../zero-shot-embed/.venv/bin/python run_scheme_failures.py
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from collections import deque
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SMALL = ROOT / "data" / "limit-small"
FULL = ROOT / "data" / "limit"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

sys.path.insert(0, str(HERE))
from glosses import GLOSSES  # noqa: E402

TOKEN_RE = re.compile(r"[a-z0-9]+")
TOKEN_RE_CS = re.compile(r"[A-Za-z0-9]+")
K1 = 1.5
B = 0.75
RRF_K = 60
SEED = 0
ADDED = (0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000)
STOP = {
    "a", "an", "the", "of", "and", "or", "to", "for", "in", "on", "with",
    "from", "who", "likes", "enjoys", "person", "by", "as", "at", "is", "its",
    "their", "that", "this", "into", "over", "than", "other", "not",
}
QUERY_STOP = {
    "who", "likes", "enjoys", "a", "an", "the", "of", "and", "or", "to",
    "for", "in", "on", "with", "from", "person", "that", "is", "known",
    "something", "this", "by", "at", "into", "over",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
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


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in STOP and len(t) > 1}


def split_likes(text: str) -> tuple[str, list[str]]:
    """Split 'Name likes A, B and C.' Items may themselves contain ' and '."""
    stripped = text.strip()
    match = re.match(r"^(.*?) likes (.*)\.\s*$", stripped)
    if not match:
        raise ValueError(f"unexpected document: {stripped[:80]!r}")
    name, rest = match.group(1), match.group(2)
    parts = [part.strip() for part in rest.split(", ") if part.strip()]
    if parts and " and " in parts[-1]:
        left, right = parts[-1].rsplit(" and ", 1)
        parts = parts[:-1] + [left.strip(), right.strip()]
    attrs = [part for part in parts if part]
    if not attrs:
        raise ValueError(f"no attributes in {stripped[:80]!r}")
    return name, attrs


def split_likes_legacy(text: str) -> tuple[str, list[str]]:
    """The splitter used by experiments/unified-score/chunk_maxp.py.

    It breaks likes-items that contain the word 'and' (Fish and Chips).
    """
    match = re.match(r"^(.*?) likes (.*)\.\s*$", text.strip())
    if not match:
        return text, [text]
    name, rest = match.group(1), match.group(2)
    parts = re.split(r",\s*|\s+and\s+", rest)
    return name, [part.strip() for part in parts if part.strip()]


def render(name: str, attrs: list[str]) -> str:
    if len(attrs) == 1:
        body = attrs[0]
    elif len(attrs) == 2:
        body = f"{attrs[0]} and {attrs[1]}"
    else:
        body = ", ".join(attrs[:-1]) + " and " + attrs[-1]
    return f"{name} likes {body}."


def attribute_of(query_text: str) -> str:
    prefix = "Who likes "
    if not (query_text.startswith(prefix) and query_text.endswith("?")):
        raise ValueError(f"unexpected query: {query_text!r}")
    return query_text[len(prefix) : -1]


def lucene_idf(n_docs: int, df: int) -> float:
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def build_aho(patterns: list[str]):
    goto: list[dict[str, int]] = [{}]
    out: list[list[int]] = [[]]
    for index, pattern in enumerate(patterns):
        node = 0
        for ch in pattern:
            nxt = goto[node].get(ch)
            if nxt is None:
                nxt = len(goto)
                goto[node][ch] = nxt
                goto.append({})
                out.append([])
            node = nxt
        out[node].append(index)
    fail = [0] * len(goto)
    queue: deque[int] = deque()
    for nxt in goto[0].values():
        queue.append(nxt)
    while queue:
        node = queue.popleft()
        for ch, nxt in list(goto[node].items()):
            queue.append(nxt)
            state = fail[node]
            while state and ch not in goto[state]:
                state = fail[state]
            fail[nxt] = goto[state].get(ch, 0)
            if fail[nxt] == nxt:
                fail[nxt] = 0
            merged = out[nxt] + out[fail[nxt]]
            out[nxt] = list(dict.fromkeys(merged))
    return goto, fail, out


def aho_find(text: str, automaton) -> list[int]:
    goto, fail, out = automaton
    found: list[int] = []
    node = 0
    for ch in text:
        while node and ch not in goto[node]:
            node = fail[node]
        node = goto[node].get(ch, 0)
        if out[node]:
            found.extend(out[node])
    return list(dict.fromkeys(found))


class BM25:
    def __init__(self, texts: list[str], doc_ids: list[str]):
        self.doc_ids = doc_ids
        self.doc_len = np.asarray([len(tokenize(text)) for text in texts], dtype=np.float64)
        postings: dict[str, list[int]] = {}
        tfs: dict[str, list[float]] = {}
        for doc_index, text in enumerate(texts):
            counts: dict[str, int] = {}
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1
            for token, tf in counts.items():
                postings.setdefault(token, []).append(doc_index)
                tfs.setdefault(token, []).append(float(tf))
        self.postings = {token: np.asarray(idx, dtype=np.int32) for token, idx in postings.items()}
        self.tfs = {token: np.asarray(tf, dtype=np.float64) for token, tf in tfs.items()}
        self.n_docs = len(texts)
        self.avgdl = float(self.doc_len.mean()) if len(texts) else 1.0
        self.idf = {
            token: lucene_idf(self.n_docs, int(idx.shape[0]))
            for token, idx in self.postings.items()
        }

    def score(self, queries: list[str]) -> np.ndarray:
        scores = np.zeros((len(queries), self.n_docs), dtype=np.float64)
        for row, query in enumerate(queries):
            seen: set[str] = set()
            for token in tokenize(query):
                if token in seen or token not in self.idf:
                    continue
                seen.add(token)
                idx = self.postings[token]
                tf = self.tfs[token]
                norm = 1.0 - B + B * self.doc_len[idx] / self.avgdl
                denom = tf + K1 * norm
                scores[row, idx] += self.idf[token] * (tf * (K1 + 1.0)) / denom
        return scores


def id_key(doc_ids: list[str]) -> np.ndarray:
    order = np.argsort(np.asarray(doc_ids), kind="mergesort")
    key = np.empty(len(doc_ids), dtype=np.int32)
    key[order] = np.arange(len(doc_ids), dtype=np.int32)
    return key


def gold_ranks(scores: np.ndarray, gold_index: np.ndarray, key: np.ndarray) -> np.ndarray:
    """1-indexed ranks. Higher score wins; equal scores break toward the earlier id."""
    n_queries, n_gold = gold_index.shape
    ranks = np.empty((n_queries, n_gold), dtype=np.int32)
    g_scores = np.take_along_axis(scores, gold_index, axis=1)
    g_keys = key[gold_index]
    for start in range(0, n_queries, 20):
        end = min(start + 20, n_queries)
        sl = scores[start:end]
        gs = g_scores[start:end]
        gk = g_keys[start:end]
        higher = (sl[:, None, :] > gs[:, :, None]).sum(axis=-1)
        earlier_tie = (
            (sl[:, None, :] == gs[:, :, None]) & (key[None, None, :] < gk[:, :, None])
        ).sum(axis=-1)
        ranks[start:end] = higher + earlier_tie + 1
    return ranks


def pack_metrics(ranks: np.ndarray) -> dict:
    n_queries, n_gold = ranks.shape
    out = {}
    for k in (2, 10, 20):
        hits = int((ranks <= k).sum())
        out[f"recall@{k}"] = round(100.0 * hits / (n_queries * n_gold), 2)
    out["both-in-top2"] = round(100.0 * float((ranks <= 2).all(axis=1).mean()), 2)
    better = ranks.min(axis=1)
    out["mean-better-gold-rank"] = round(float(better.mean()), 2)
    out["median-better-gold-rank"] = round(float(np.median(better)), 2)
    return out


def metrics_from_scores(scores: np.ndarray, gold_index: np.ndarray, key: np.ndarray) -> dict:
    return pack_metrics(gold_ranks(scores, gold_index, key))


def full_ranks(scores: np.ndarray, key: np.ndarray) -> np.ndarray:
    """0-indexed rank of every document. Rank 0 is the top."""
    n_queries, n_docs = scores.shape
    ranks = np.empty((n_queries, n_docs), dtype=np.int32)
    for row in range(n_queries):
        order = np.lexsort((key, -scores[row]))
        ranks[row, order] = np.arange(n_docs, dtype=np.int32)
    return ranks


def fusion_block(scores_a: np.ndarray, scores_b: np.ndarray, gold_index: np.ndarray,
                 key: np.ndarray, name_a: str, name_b: str) -> dict:
    """Union ceiling and RRF on a shared document set."""
    ranks_a = full_ranks(scores_a, key)
    ranks_b = full_ranks(scores_b, key)
    rrf = 1.0 / (RRF_K + ranks_a + 1) + 1.0 / (RRF_K + ranks_b + 1)
    gold_a = np.take_along_axis(ranks_a, gold_index, axis=1) + 1
    gold_b = np.take_along_axis(ranks_b, gold_index, axis=1) + 1
    out = {
        "channels": [name_a, name_b],
        "a": pack_metrics(gold_a),
        "b": pack_metrics(gold_b),
        "rrf-full-list": metrics_from_scores(rrf, gold_index, key),
    }
    n_queries, n_gold = gold_index.shape
    slots = n_queries * n_gold
    for k in (2, 10, 20):
        miss_a = gold_a > k
        miss_b = gold_b > k
        both = miss_a & miss_b
        either = ~(both)
        # Queries where neither channel places either gold in the top k.
        query_both_blind = both.all(axis=1)
        out[f"k={k}"] = {
            "union-recall": round(100.0 * float(either.sum()) / slots, 2),
            "both-miss-gold-slots": int(both.sum()),
            "both-miss-gold-slot-rate": round(100.0 * float(both.sum()) / slots, 2),
            "queries-both-channels-miss-both-golds": int(query_both_blind.sum()),
        }
        # RRF restricted to the union of the two top-k lists. Documents outside
        # the union are invisible, which is the late-fusion ceiling.
        visible = (ranks_a < k) | (ranks_b < k)
        truncated = np.where(visible, rrf, -1e9)
        trunc_ranks = gold_ranks(truncated, gold_index, key)
        # An invisible gold must not fill a spare top-k slot when the union is
        # shorter than k and the remaining scores tie at -1e9.
        visible_gold = np.take_along_axis(visible, gold_index, axis=1)
        trunc_ranks = trunc_ranks.copy()
        trunc_ranks[~visible_gold] = 10**6
        out[f"k={k}"]["rrf-on-union-top"] = pack_metrics(trunc_ranks)
    return out


def reduce_chunks(q_emb: np.ndarray, c_emb: np.ndarray, owners: np.ndarray,
                  n_docs: int, how: str) -> np.ndarray:
    token = q_emb @ c_emb.T
    scores = np.full((q_emb.shape[0], n_docs), -1e9, dtype=np.float32)
    for doc_index in range(n_docs):
        mask = owners == doc_index
        if not np.any(mask):
            continue
        block = token[:, mask]
        scores[:, doc_index] = block.max(axis=1) if how == "max" else block.mean(axis=1)
    return scores


def group_chunks(parsed: list[tuple[str, list[str]]], k: int | None) -> tuple[list[str], np.ndarray]:
    texts: list[str] = []
    owners: list[int] = []
    for doc_index, (name, attrs) in enumerate(parsed):
        width = len(attrs) if k is None else k
        for start in range(0, len(attrs), width):
            texts.append(render(name, attrs[start : start + width]))
            owners.append(doc_index)
    return texts, np.asarray(owners, dtype=np.int32)


def window_chunks(texts: list[str], size: int, stride: int, offset: int) -> tuple[list[str], np.ndarray, list[list[str]]]:
    chunk_texts: list[str] = []
    owners: list[int] = []
    token_windows: list[list[str]] = []
    for doc_index, text in enumerate(texts):
        tokens = TOKEN_RE_CS.findall(text)
        start = min(offset, max(len(tokens) - 1, 0))
        if start >= len(tokens):
            continue
        pos = start
        while pos < len(tokens):
            piece = tokens[pos : pos + size]
            if not piece:
                break
            token_windows.append(piece)
            chunk_texts.append(" ".join(piece))
            owners.append(doc_index)
            if stride <= 0:
                break
            pos += stride
            if len(piece) < size:
                break
    return chunk_texts, np.asarray(owners, dtype=np.int32), token_windows


def contains_seq(haystack: list[str], needle: list[str]) -> bool:
    width = len(needle)
    if width == 0 or width > len(haystack):
        return False
    for start in range(len(haystack) - width + 1):
        if haystack[start : start + width] == needle:
            return True
    return False


def containment_rate(window_tokens: list[list[str]], owners: np.ndarray,
                     parsed: list[tuple[str, list[str]]], query_attrs: list[str],
                     gold_index: np.ndarray) -> dict:
    """Fraction of (query, gold doc) pairs whose attribute is intact in some window."""
    by_doc: list[list[list[str]]] = [[] for _ in parsed]
    for tokens, owner in zip(window_tokens, owners):
        by_doc[int(owner)].append(tokens)
    needles = [TOKEN_RE_CS.findall(attr) for attr in query_attrs]
    hit = 0
    total = 0
    for row, needle in enumerate(needles):
        for slot in range(gold_index.shape[1]):
            total += 1
            doc_index = int(gold_index[row, slot])
            if any(contains_seq(tokens, needle) for tokens in by_doc[doc_index]):
                hit += 1
    return {"gold-spans-intact": hit, "gold-spans": total,
            "intact-rate": round(100.0 * hit / total, 2) if total else 0.0}


def validate_glosses(glosses: dict[str, str]) -> list[dict]:
    bad = []
    for attr, gloss in glosses.items():
        overlap = sorted(content_tokens(attr) & content_tokens(gloss))
        substring = attr.lower() in gloss.lower()
        if overlap or substring:
            bad.append({"attribute": attr, "gloss": gloss, "overlap": overlap,
                        "attribute-substring-of-gloss": substring})
    return bad


def attr_char_spans(text: str, attrs: list[str]) -> list[tuple[int, int]]:
    cursor = text.find(" likes ")
    if cursor < 0:
        raise ValueError("no likes marker")
    cursor += len(" likes ")
    spans = []
    for attr in attrs:
        start = text.find(attr, cursor)
        if start < 0 or text[start : start + len(attr)] != attr:
            raise ValueError(f"span miss for {attr!r}")
        spans.append((start, start + len(attr)))
        cursor = start + len(attr)
    return spans


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def encode_sentences(model, texts: list[str], batch_size: int = 256) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def token_packs(model, texts: list[str], drop_query_stops: bool) -> tuple[np.ndarray, np.ndarray]:
    """Concatenated L2-normalized token vectors and a query/doc owner per row."""
    embeddings = model.encode(
        texts,
        batch_size=32,
        output_value="token_embeddings",
        normalize_embeddings=False,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    embeddings = [as_numpy(item) for item in embeddings]
    tokenizer = model.tokenizer
    rows: list[np.ndarray] = []
    owners: list[int] = []
    special = {"[CLS]", "[SEP]", "[PAD]"}
    punct = {",", ".", "?", "!", ";", ":", "'", '"'}
    for index, (text, emb) in enumerate(zip(texts, embeddings)):
        ids = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=256,
        )["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(ids)
        if len(tokens) != len(emb):
            width = min(len(tokens), len(emb))
            tokens = tokens[:width]
            emb = emb[:width]
        keep = []
        for token_index, token in enumerate(tokens):
            if token in special or token in punct:
                continue
            surface = token[2:] if token.startswith("##") else token
            surface = surface.lower()
            if drop_query_stops and not token.startswith("##") and surface in QUERY_STOP:
                continue
            keep.append(token_index)
        if not keep:
            keep = [i for i, token in enumerate(tokens) if token not in special] or [0]
        vecs = np.asarray(emb[keep], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-8)
        rows.append(vecs)
        owners.extend([index] * len(vecs))
    packed = np.vstack(rows)
    return packed, np.asarray(owners, dtype=np.int32)


def maxsim_scores(q_vecs: np.ndarray, q_owner: np.ndarray, d_vecs: np.ndarray,
                  d_owner: np.ndarray, n_queries: int, n_docs: int, how: str) -> np.ndarray:
    scores = np.zeros((n_queries, n_docs), dtype=np.float32)
    doc_slices = []
    for doc_index in range(n_docs):
        positions = np.flatnonzero(d_owner == doc_index)
        doc_slices.append(d_vecs[positions])
    for doc_index, d_mat in enumerate(doc_slices):
        sims = q_vecs @ d_mat.T
        best = sims.max(axis=1)
        if how == "sum":
            scores[:, doc_index] = np.bincount(q_owner, weights=best, minlength=n_queries)
        else:
            # max over query tokens, not the sum. One strong token can carry the doc.
            order = np.argsort(q_owner, kind="mergesort")
            grouped_owner = q_owner[order]
            grouped_best = best[order]
            # reduceat needs contiguous groups; argsort gives that.
            cuts = np.flatnonzero(np.diff(grouped_owner)) + 1
            starts = np.r_[0, cuts]
            reduced = np.maximum.reduceat(grouped_best, starts)
            scores[grouped_owner[starts], doc_index] = reduced
    return scores


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def write_report(res: dict, path: Path) -> None:
    lines = [
        "# Scheme failure measurements (LIMIT-small, MiniLM-L6-v2)",
        "",
        "Recall is macro-averaged `|top-k ∩ gold| / 2` on a 0–100 scale. "
        "Ties break toward the alphabetically earlier document id. "
        "`data/limit` distractors are a different draw (seed-0 permutation), not the official negatives.",
        "",
        "## Reproduction",
        "",
        "| scoring | R@2 | R@10 | R@20 | both-in-top2 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in ("whole-doc", "chunk-max", "chunk-avg", "legacy-split-chunk-max", "bm25"):
        row = res["reproduction"][label]
        lines.append(
            f"| {label} | {row['recall@2']} | {row['recall@10']} | {row['recall@20']} | {row['both-in-top2']} |"
        )
    lines += ["", "## Query wording, all 1000 queries, documents unchanged", ""]
    lines.append("| query | BM25 R@2 | whole-doc R@2 | chunk-MaxP R@2 | chunk-AvgP R@2 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for label, row in res["query-templates"].items():
        lines.append(
            f"| {label} | {row['bm25']['recall@2']} | {row['whole-doc']['recall@2']} | "
            f"{row['chunk-max']['recall@2']} | {row['chunk-avg']['recall@2']} |"
        )
    lines += ["", "## Glosses that share no content token with the attribute", ""]
    g = res["gloss"]
    lines.append(
        f"Subset size {g['n']}. Strict lexical cutoff after doc rewrite: "
        f"{g['strict-lexical-cutoff-n']} queries whose original content tokens "
        f"have document frequency 0."
    )
    lines.append("")
    lines.append("| setting | n | BM25 R@2 | whole-doc R@2 | chunk-MaxP R@2 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for label, row in g["settings"].items():
        lines.append(
            f"| {label} | {row['n']} | {row['bm25']['recall@2']} | "
            f"{row['whole-doc']['recall@2']} | {row['chunk-max']['recall@2']} |"
        )
    lines += ["", "## Chunk geometry", ""]
    lines.append("| chunks | intact gold spans | R@2 | R@10 | both-in-top2 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for label, row in res["chunk-geometry"].items():
        intact = row.get("intact-rate", "—")
        lines.append(
            f"| {label} | {intact} | {row['recall@2']} | {row['recall@10']} | {row['both-in-top2']} |"
        )
    lines += ["", "## Late interaction proxy", ""]
    for label, row in res["late-interaction"].items():
        if isinstance(row, dict) and "recall@2" in row:
            lines.append(
                f"- {label}: R@2 {row['recall@2']}, R@10 {row['recall@10']}, "
                f"both-in-top2 {row['both-in-top2']}"
            )
        else:
            lines.append(f"- {label}: {row}")
    lines += ["", "## Fusion ceiling (RRF k=60)", ""]
    for label, block in res["fusion"].items():
        k2 = block["k=2"]
        lines.append(
            f"- {label}: {block['channels'][0]} R@2 {block['a']['recall@2']}, "
            f"{block['channels'][1]} R@2 {block['b']['recall@2']}, "
            f"union R@2 {k2['union-recall']}, both-miss slots {k2['both-miss-gold-slots']}, "
            f"queries blind at 2: {k2['queries-both-channels-miss-both-golds']}, "
            f"full-list RRF R@2 {block['rrf-full-list']['recall@2']}, "
            f"RRF-on-union R@2 {k2['rrf-on-union-top']['recall@2']}"
        )
    if "drowning" in res:
        lines += ["", "## Drowning (LIMIT-small golds, distractors from data/limit)", ""]
        lines.append("| added | whole R@2 | whole semantic-only R@2 | MaxP R@2 | MaxP semantic-only R@2 | MaxP exact-string-only R@2 | BM25 R@2 |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in res["drowning"]["curve"]:
            lines.append(
                f"| {row['added']} | {row['whole-doc']['recall@2']} | "
                f"{row['whole-semantic-only']['recall@2']} | {row['chunk-max']['recall@2']} | "
                f"{row['chunk-max-semantic-only']['recall@2']} | "
                f"{row['chunk-max-exact-string-only']['recall@2']} | {row['bm25']['recall@2']} |"
            )
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def subset_rows(indices: np.ndarray, *arrays: np.ndarray) -> list[np.ndarray]:
    return [arr[indices] for arr in arrays]


def main() -> None:
    t0 = time.time()
    bad_glosses = validate_glosses(GLOSSES)
    if bad_glosses:
        raise SystemExit(f"glosses share tokens with attributes: {bad_glosses}")

    automaton = build_aho(["Ham", "Hamburgers", "Horses", "Horseshoe"])
    demo = aho_find("Flora Hammaker likes Hamburgers and Horseshoe Crabs.", automaton)
    demo_attrs = [a for i, a in enumerate(["Ham", "Hamburgers", "Horses", "Horseshoe"]) if i in demo]
    if demo_attrs != ["Ham", "Hamburgers", "Horses", "Horseshoe"]:
        raise SystemExit(f"Aho demo failed: {demo_attrs}")

    small_docs = load_jsonl(SMALL / "corpus.jsonl")
    queries = load_jsonl(SMALL / "queries.jsonl")
    qrels = load_jsonl(SMALL / "qrels.jsonl")
    gold_map: dict[str, list[str]] = {}
    for row in qrels:
        gold_map.setdefault(row["query-id"], []).append(row["corpus-id"])

    small_ids = [row["_id"] for row in small_docs]
    small_texts = [doc_text(row) for row in small_docs]
    parsed = [split_likes(text) for text in small_texts]
    for text, (name, attrs) in zip(small_texts, parsed):
        if render(name, attrs) != text.strip():
            raise SystemExit(f"splitter did not round-trip {name}")

    query_ids = [row["_id"] for row in queries]
    query_texts = [row["text"] for row in queries]
    query_attrs = [attribute_of(text) for text in query_texts]
    if len(set(query_attrs)) != len(query_attrs):
        raise SystemExit("expected unique attributes")
    gold_index = []
    id_to_pos = {doc_id: i for i, doc_id in enumerate(small_ids)}
    for qid in query_ids:
        golds = gold_map[qid]
        if len(golds) != 2 or any(g not in id_to_pos for g in golds):
            raise SystemExit(f"bad golds for {qid}")
        gold_index.append([id_to_pos[g] for g in golds])
    gold_index = np.asarray(gold_index, dtype=np.int32)
    small_key = id_key(small_ids)

    gloss_rows = [i for i, attr in enumerate(query_attrs) if attr in GLOSSES]
    gloss_index = np.asarray(gloss_rows, dtype=np.int32)
    missing = sorted(set(GLOSSES) - set(query_attrs))
    if missing:
        raise SystemExit(f"glosses not in the query file: {missing[:5]}")

    print(f"loaded small docs={len(small_ids)} queries={len(query_ids)} glosses={len(gloss_index)}", flush=True)

    # --- lexical channel on the 46 ---
    bm25 = BM25(small_texts, small_ids)
    template_queries = {
        "original": query_texts,
        "enjoys": [f"Who enjoys {attr}?" for attr in query_attrs],
        "person-enjoys": [f"A person who enjoys {attr}" for attr in query_attrs],
        "attr-only": list(query_attrs),
        "drop-attribute": ["Who likes something?" for _ in query_attrs],
    }
    bm25_scores = {name: bm25.score(texts).astype(np.float32) for name, texts in template_queries.items()}

    rewritten_attrs = []
    rewritten_texts = []
    for name, attrs in parsed:
        replaced = [GLOSSES.get(attr, attr) for attr in attrs]
        rewritten_attrs.append(replaced)
        rewritten_texts.append(render(name, replaced))
    bm25_rewritten = BM25(rewritten_texts, small_ids)
    gloss_query_texts = [f"Who likes {GLOSSES[query_attrs[i]]}?" for i in gloss_index]
    gloss_only_texts = [GLOSSES[query_attrs[i]] for i in gloss_index]

    # Residual lexical hits: original attribute content tokens still in some rewritten doc.
    rewritten_token_sets = [set(tokenize(text)) for text in rewritten_texts]
    strict_rows = []
    residual_df = []
    for row in gloss_index.tolist():
        tokens = content_tokens(query_attrs[row])
        dfs = []
        alive = False
        for token in tokens:
            df = sum(token in toks for toks in rewritten_token_sets)
            dfs.append(df)
            if df > 0:
                alive = True
        residual_df.append(0 if not dfs else min(dfs))
        if not alive:
            strict_rows.append(row)
    strict_index = np.asarray(strict_rows, dtype=np.int32)

    import torch
    from sentence_transformers import SentenceTransformer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    model = SentenceTransformer(MODEL_NAME, device=device)
    pooling_name = type(model[1]).__name__ if len(model) > 1 else "none"

    # Sanity: official sentence vector vs the pooling module on all tokens.
    probe = "Who likes Horses?"
    probe_sentence = encode_sentences(model, [probe])
    probe_tokens = as_numpy(model.encode(
        [probe], output_value="token_embeddings", normalize_embeddings=False,
        show_progress_bar=False, convert_to_numpy=True,
    )[0])
    probe_tensor = torch.tensor(probe_tokens, device=device).unsqueeze(0)
    probe_mask = torch.ones(probe_tensor.shape[:2], device=device)
    with torch.no_grad():
        pooled = model[1]({"token_embeddings": probe_tensor, "attention_mask": probe_mask})
        pooled_vec = pooled["sentence_embedding"].detach().float().cpu().numpy()[0]
    pooled_vec = pooled_vec / max(float(np.linalg.norm(pooled_vec)), 1e-8)
    pool_agreement = float(np.dot(pooled_vec, probe_sentence[0]))
    print(f"pooling={pooling_name} agreement={pool_agreement:.4f}", flush=True)

    all_query_texts = []
    query_slices = {}
    for name, texts in list(template_queries.items()) + [
        ("gloss-who-likes", gloss_query_texts),
        ("gloss-only", gloss_only_texts),
    ]:
        query_slices[name] = (len(all_query_texts), len(all_query_texts) + len(texts))
        all_query_texts.extend(texts)
    q_all = encode_sentences(model, all_query_texts, batch_size=256)
    q_bank = {name: q_all[start:end] for name, (start, end) in query_slices.items()}

    d_small = encode_sentences(model, small_texts, batch_size=64)
    d_rewritten = encode_sentences(model, rewritten_texts, batch_size=64)

    chunk_jobs: dict[str, tuple[list[str], np.ndarray]] = {}
    chunk_jobs["k1"] = group_chunks(parsed, 1)
    for width in (2, 4, 8, 16):
        chunk_jobs[f"k{width}"] = group_chunks(parsed, width)
    chunk_jobs["all"] = group_chunks(parsed, None)
    legacy_parsed = [split_likes_legacy(text) for text in small_texts]
    legacy_texts, legacy_owners = [], []
    for doc_index, (name, attrs) in enumerate(legacy_parsed):
        for attr in attrs:
            legacy_texts.append(f"{name} likes {attr}.")
            legacy_owners.append(doc_index)
    chunk_jobs["legacy-k1"] = (legacy_texts, np.asarray(legacy_owners, dtype=np.int32))
    chunk_jobs["glossed-k1"] = group_chunks(
        [(name, attrs) for (name, _), attrs in zip(parsed, rewritten_attrs)], 1
    )

    window_meta = {}
    for size, stride, offset, label in (
        (8, 4, 0, "win8-stride4"),
        (8, 4, 3, "win8-stride4-offset3"),
        (16, 8, 0, "win16-stride8"),
        (32, 16, 0, "win32-stride16"),
        (10, 10, 0, "win10-nongap"),
    ):
        texts, owners, token_windows = window_chunks(small_texts, size, stride, offset)
        chunk_jobs[label] = (texts, owners)
        window_meta[label] = (token_windows, owners)
    rng = np.random.default_rng(SEED)
    random_texts, random_owners, random_tokens = [], [], []
    for doc_index, text in enumerate(small_texts):
        tokens = TOKEN_RE_CS.findall(text)
        start = int(rng.integers(0, 8)) if len(tokens) > 8 else 0
        pos = start
        while pos < len(tokens):
            piece = tokens[pos : pos + 10]
            if not piece:
                break
            random_tokens.append(piece)
            random_texts.append(" ".join(piece))
            random_owners.append(doc_index)
            pos += 10
    random_owners_arr = np.asarray(random_owners, dtype=np.int32)
    chunk_jobs["random-win10"] = (random_texts, random_owners_arr)
    window_meta["random-win10"] = (random_tokens, random_owners_arr)

    flat_texts: list[str] = []
    flat_slices = {}
    for label, (texts, _) in chunk_jobs.items():
        flat_slices[label] = (len(flat_texts), len(flat_texts) + len(texts))
        flat_texts.extend(texts)
    print(f"encoding {len(flat_texts)} small-corpus chunks", flush=True)
    flat_emb = encode_sentences(model, flat_texts, batch_size=512)
    c_bank = {label: flat_emb[start:end] for label, (start, end) in flat_slices.items()}

    def chunk_scores(q_name: str, c_name: str, how: str) -> np.ndarray:
        return reduce_chunks(q_bank[q_name], c_bank[c_name], chunk_jobs[c_name][1], len(small_ids), how)

    reproduction = {
        "whole-doc": metrics_from_scores(q_bank["original"] @ d_small.T, gold_index, small_key),
        "chunk-max": metrics_from_scores(chunk_scores("original", "k1", "max"), gold_index, small_key),
        "chunk-avg": metrics_from_scores(chunk_scores("original", "k1", "avg"), gold_index, small_key),
        "legacy-split-chunk-max": metrics_from_scores(
            chunk_scores("original", "legacy-k1", "max"), gold_index, small_key
        ),
        "bm25": metrics_from_scores(bm25_scores["original"], gold_index, small_key),
    }
    print("reproduction", json.dumps(reproduction), flush=True)

    query_template_metrics = {}
    for name in template_queries:
        query_template_metrics[name] = {
            "bm25": metrics_from_scores(bm25_scores[name], gold_index, small_key),
            "whole-doc": metrics_from_scores(q_bank[name] @ d_small.T, gold_index, small_key),
            "chunk-max": metrics_from_scores(chunk_scores(name, "k1", "max"), gold_index, small_key),
            "chunk-avg": metrics_from_scores(chunk_scores(name, "k1", "avg"), gold_index, small_key),
        }
    print("templates", {k: {m: v[m]["recall@2"] for m in v} for k, v in query_template_metrics.items()}, flush=True)

    # Geometry curve on original queries.
    geometry = {}
    for label in ("k1", "k2", "k4", "k8", "k16", "all", "win8-stride4", "win8-stride4-offset3",
                  "win16-stride8", "win32-stride16", "win10-nongap", "random-win10"):
        scores = chunk_scores("original", label, "max")
        geometry[label] = metrics_from_scores(scores, gold_index, small_key)
        geometry[label]["n-chunks"] = int(chunk_jobs[label][1].shape[0])
        if label in window_meta:
            tokens, owners = window_meta[label]
            geometry[label].update(containment_rate(tokens, owners, parsed, query_attrs, gold_index))
        elif label == "k1":
            geometry[label]["intact-rate"] = 100.0
        elif label == "all":
            geometry[label]["intact-rate"] = 100.0
        else:
            # Grouped likes-items keep each attribute whole inside exactly one chunk.
            geometry[label]["intact-rate"] = 100.0
    geometry["whole-doc"] = dict(reproduction["whole-doc"])
    geometry["whole-doc"]["intact-rate"] = 100.0
    geometry["whole-doc"]["n-chunks"] = len(small_ids)
    geometry["chunk-avg-k1"] = dict(reproduction["chunk-avg"])
    geometry["chunk-avg-k1"]["intact-rate"] = 100.0
    geometry["chunk-avg-k1"]["n-chunks"] = int(chunk_jobs["k1"][1].shape[0])

    # --- gloss subset ---
    def eval_subset(rows: np.ndarray, bm25_mat: np.ndarray, dense_mat: np.ndarray,
                    chunk_mat: np.ndarray) -> dict:
        if len(rows) == 0:
            empty = {"recall@2": None, "recall@10": None, "recall@20": None, "both-in-top2": None}
            return {"n": 0, "bm25": empty, "whole-doc": empty, "chunk-max": empty}
        sub_gold = gold_index[rows]
        return {
            "n": int(len(rows)),
            "bm25": metrics_from_scores(bm25_mat[rows], sub_gold, small_key),
            "whole-doc": metrics_from_scores(dense_mat[rows], sub_gold, small_key),
            "chunk-max": metrics_from_scores(chunk_mat[rows], sub_gold, small_key),
        }

    bm25_gloss_q = bm25.score(gloss_query_texts).astype(np.float32)
    bm25_gloss_only = bm25.score(gloss_only_texts).astype(np.float32)
    dense_gloss_q = q_bank["gloss-who-likes"] @ d_small.T
    dense_gloss_only = q_bank["gloss-only"] @ d_small.T
    chunk_gloss_q = reduce_chunks(q_bank["gloss-who-likes"], c_bank["k1"], chunk_jobs["k1"][1], len(small_ids), "max")
    chunk_gloss_only = reduce_chunks(q_bank["gloss-only"], c_bank["k1"], chunk_jobs["k1"][1], len(small_ids), "max")

    bm25_orig_on_rewritten = bm25_rewritten.score(query_texts).astype(np.float32)
    bm25_both = bm25_rewritten.score(gloss_query_texts).astype(np.float32)
    dense_orig_on_rewritten = q_bank["original"] @ d_rewritten.T
    dense_both = q_bank["gloss-who-likes"] @ d_rewritten.T
    chunk_orig_on_rewritten = reduce_chunks(
        q_bank["original"], c_bank["glossed-k1"], chunk_jobs["glossed-k1"][1], len(small_ids), "max"
    )
    chunk_both = reduce_chunks(
        q_bank["gloss-who-likes"], c_bank["glossed-k1"], chunk_jobs["glossed-k1"][1], len(small_ids), "max"
    )

    gloss_settings = {
        "original-wording": eval_subset(
            gloss_index,
            bm25_scores["original"],
            q_bank["original"] @ d_small.T,
            chunk_scores("original", "k1", "max"),
        ),
        "query-gloss-docs-exact": {
            "n": int(len(gloss_index)),
            "bm25": metrics_from_scores(bm25_gloss_q, gold_index[gloss_index], small_key),
            "whole-doc": metrics_from_scores(dense_gloss_q, gold_index[gloss_index], small_key),
            "chunk-max": metrics_from_scores(chunk_gloss_q, gold_index[gloss_index], small_key),
        },
        "query-gloss-only-docs-exact": {
            "n": int(len(gloss_index)),
            "bm25": metrics_from_scores(bm25_gloss_only, gold_index[gloss_index], small_key),
            "whole-doc": metrics_from_scores(dense_gloss_only, gold_index[gloss_index], small_key),
            "chunk-max": metrics_from_scores(chunk_gloss_only, gold_index[gloss_index], small_key),
        },
        "query-exact-docs-gloss": eval_subset(
            gloss_index, bm25_orig_on_rewritten, dense_orig_on_rewritten, chunk_orig_on_rewritten
        ),
        "both-gloss": {
            "n": int(len(gloss_index)),
            "bm25": metrics_from_scores(bm25_both, gold_index[gloss_index], small_key),
            "whole-doc": metrics_from_scores(dense_both, gold_index[gloss_index], small_key),
            "chunk-max": metrics_from_scores(chunk_both, gold_index[gloss_index], small_key),
        },
    }
    if len(strict_index):
        # strict_index indexes the full query list. Map into gloss-subset rows
        # for matrices that are already subset-ordered.
        strict_pos = np.asarray([gloss_rows.index(int(r)) for r in strict_index], dtype=np.int32)
        gloss_settings["strict-query-gloss-docs-exact"] = {
            "n": int(len(strict_index)),
            "bm25": metrics_from_scores(bm25_gloss_q[strict_pos], gold_index[strict_index], small_key),
            "whole-doc": metrics_from_scores(dense_gloss_q[strict_pos], gold_index[strict_index], small_key),
            "chunk-max": metrics_from_scores(chunk_gloss_q[strict_pos], gold_index[strict_index], small_key),
        }
        gloss_settings["strict-query-exact-docs-gloss"] = eval_subset(
            strict_index, bm25_orig_on_rewritten, dense_orig_on_rewritten, chunk_orig_on_rewritten
        )
        gloss_settings["strict-both-gloss"] = {
            "n": int(len(strict_index)),
            "bm25": metrics_from_scores(bm25_both[strict_pos], gold_index[strict_index], small_key),
            "whole-doc": metrics_from_scores(dense_both[strict_pos], gold_index[strict_index], small_key),
            "chunk-max": metrics_from_scores(chunk_both[strict_pos], gold_index[strict_index], small_key),
        }
    print("gloss", {k: (v["n"], v["bm25"]["recall@2"], v["whole-doc"]["recall@2"], v["chunk-max"]["recall@2"])
                    for k, v in gloss_settings.items()}, flush=True)

    # Per-gloss cosine margins for the note.
    k1_owners = chunk_jobs["k1"][1]
    glossed_owners = chunk_jobs["glossed-k1"][1]
    # Chunk i of doc corresponds to attribute i because group_chunks walks attrs in order
    # and every doc uses width 1. Confirm equal counts.
    per_doc_counts = np.bincount(k1_owners, minlength=len(small_ids))
    margins = []
    cursor = np.zeros(len(small_ids), dtype=np.int32)
    # Build attr -> list of (doc, chunk_row)
    attr_chunk_rows: dict[str, list[tuple[int, int]]] = {}
    for doc_index, (_, attrs) in enumerate(parsed):
        for attr in attrs:
            # owners were appended in the same order
            attr_chunk_rows.setdefault(attr, []).append((doc_index, None))
    # Fill chunk rows by replaying group order.
    replay = [[] for _ in small_ids]
    for row_index, doc_index in enumerate(k1_owners.tolist()):
        replay[doc_index].append(row_index)
    attr_chunk_rows = {}
    for doc_index, (_, attrs) in enumerate(parsed):
        if len(replay[doc_index]) != len(attrs):
            raise SystemExit("chunk alignment mismatch")
        for attr, row_index in zip(attrs, replay[doc_index]):
            attr_chunk_rows.setdefault(attr, []).append((doc_index, row_index))
    replay_g = [[] for _ in small_ids]
    for row_index, doc_index in enumerate(glossed_owners.tolist()):
        replay_g[doc_index].append(row_index)
    q_orig = q_bank["original"]
    exact_chunk = c_bank["k1"]
    gloss_chunk = c_bank["glossed-k1"]
    for pos, row in enumerate(gloss_index.tolist()):
        attr = query_attrs[row]
        gold_rows = []
        gloss_rows_for_attr = []
        for doc_index, attrs in ((i, a) for i, (_, a) in enumerate(parsed)):
            pass
        for doc_index in gold_index[row].tolist():
            attrs = parsed[doc_index][1]
            which = attrs.index(attr)
            gold_rows.append(replay[doc_index][which])
            gloss_rows_for_attr.append(replay_g[doc_index][which])
        q = q_orig[row]
        qg = q_bank["gloss-who-likes"][pos]
        exact_vecs = exact_chunk[gold_rows]
        gloss_vecs = gloss_chunk[gloss_rows_for_attr]
        margins.append({
            "attribute": attr,
            "cos-orig-query-exact-chunk": round(float((exact_vecs @ q).mean()), 4),
            "cos-gloss-query-exact-chunk": round(float((exact_vecs @ qg).mean()), 4),
            "cos-orig-query-gloss-chunk": round(float((gloss_vecs @ q).mean()), 4),
            "cos-gloss-query-gloss-chunk": round(float((gloss_vecs @ qg).mean()), 4),
        })
    margin_means = {
        key: round(float(np.mean([m[key] for m in margins])), 4)
        for key in margins[0] if key != "attribute"
    }

    # --- collisions inside the 46 ---
    doc_attrs = [attrs for _, attrs in parsed]
    collision_rows = []
    for row, attr in enumerate(query_attrs):
        carriers = []
        for doc_index, attrs in enumerate(doc_attrs):
            if doc_index in set(gold_index[row].tolist()):
                continue
            for other in attrs:
                if other != attr and attr in other:
                    carriers.append({"doc": small_ids[doc_index], "attribute": other})
                    break
        if carriers:
            collision_rows.append(row)
    collision_index = np.asarray(collision_rows, dtype=np.int32)
    chunk_orig_scores = chunk_scores("original", "k1", "max")
    collision_detail = []
    chunk_ranks = gold_ranks(chunk_orig_scores, gold_index, small_key)
    bm25_ranks = gold_ranks(bm25_scores["original"], gold_index, small_key)
    whole_ranks = gold_ranks(q_bank["original"] @ d_small.T, gold_index, small_key)
    for row in collision_index.tolist():
        attr = query_attrs[row]
        carriers = []
        for doc_index, attrs in enumerate(doc_attrs):
            for other in attrs:
                if other != attr and attr in other:
                    carriers.append((doc_index, other))
        # rank of each carrier doc
        order = np.lexsort((small_key, -chunk_orig_scores[row]))
        rank_of = {int(doc): r for r, doc in enumerate(order)}
        bm_order = np.lexsort((small_key, -bm25_scores["original"][row]))
        bm_rank = {int(doc): r for r, doc in enumerate(bm_order)}
        collision_detail.append({
            "attribute": attr,
            "chunk-gold-ranks": chunk_ranks[row].tolist(),
            "bm25-gold-ranks": bm25_ranks[row].tolist(),
            "carriers": [
                {"doc": small_ids[doc], "superstring": other,
                 "chunk-rank": rank_of[doc] + 1, "bm25-rank": bm_rank[doc] + 1,
                 "is-gold": doc in set(gold_index[row].tolist())}
                for doc, other in carriers
            ],
        })

    # Chunk-MaxP misses: which chunk won?
    fail_examples = []
    k1_texts = chunk_jobs["k1"][0]
    q_chunk = q_bank["original"] @ c_bank["k1"].T
    for row in range(len(query_ids)):
        if (chunk_ranks[row] <= 2).all():
            continue
        order = np.lexsort((small_key, -chunk_orig_scores[row]))
        top_doc = int(order[0])
        rows_of_top = np.flatnonzero(k1_owners == top_doc)
        local = q_chunk[row, rows_of_top]
        best_local = int(rows_of_top[int(local.argmax())])
        best_text = k1_texts[best_local]
        attr = query_attrs[row]
        fail_examples.append({
            "query-id": query_ids[row],
            "attribute": attr,
            "gold-ranks": chunk_ranks[row].tolist(),
            "top-doc": small_ids[top_doc],
            "top-is-gold": top_doc in set(gold_index[row].tolist()),
            "best-chunk": best_text,
            "attribute-in-best-chunk": attr in best_text,
            "shared-content-tokens": sorted(content_tokens(attr) & content_tokens(best_text)),
        })

    # Token-disjoint semantic neighbors among attribute phrases.
    phrase = [f"likes {attr}." for attr in query_attrs]
    phrase_emb = encode_sentences(model, phrase, batch_size=256)
    phrase_sim = phrase_emb @ phrase_emb.T
    np.fill_diagonal(phrase_sim, -1e9)
    neighbor_cos = []
    neighbor_beats = 0
    for row, attr in enumerate(query_attrs):
        mine = content_tokens(attr)
        order = np.argsort(-phrase_sim[row])
        picked = None
        for other in order.tolist():
            if not (mine & content_tokens(query_attrs[other])):
                picked = other
                break
        if picked is None:
            continue
        neighbor_cos.append(float(phrase_sim[row, picked]))
        # Does any non-gold doc that likes the neighbor attribute land in chunk-MaxP top 2?
        neighbor_attr = query_attrs[picked]
        golds = set(gold_index[row].tolist())
        order_docs = np.lexsort((small_key, -chunk_orig_scores[row]))
        top2 = set(int(d) for d in order_docs[:2])
        owners_of_neighbor = {doc for doc, _ in attr_chunk_rows.get(neighbor_attr, [])}
        if (owners_of_neighbor - golds) & top2:
            neighbor_beats += 1
    semantic_neighbors = {
        "n": len(neighbor_cos),
        "mean-cosine": round(float(np.mean(neighbor_cos)), 4) if neighbor_cos else None,
        "median-cosine": round(float(np.median(neighbor_cos)), 4) if neighbor_cos else None,
        "p90-cosine": round(float(np.quantile(neighbor_cos, 0.9)), 4) if neighbor_cos else None,
        "neighbor-doc-in-chunk-max-top2": neighbor_beats,
        "queries": len(query_ids),
    }

    # Planted exact-string copies. Per query, t other people whose only chunk is the attribute.
    copy_ids = [f"Copy Person {i:02d}" for i in range(5)]
    copy_texts = []
    for attr in query_attrs:
        for copy_id in copy_ids:
            copy_texts.append(f"{copy_id} likes {attr}.")
    print(f"encoding {len(copy_texts)} planted copies", flush=True)
    copy_emb = encode_sentences(model, copy_texts, batch_size=512)
    copy_emb = copy_emb.reshape(len(query_ids), 5, -1)
    q_orig_unit = q_bank["original"]
    copy_scores = np.einsum("qh,qth->qt", q_orig_unit, copy_emb)
    planted = {}
    base_scores = chunk_orig_scores
    for t in (1, 2, 5):
        ranks = np.empty_like(gold_index)
        gaps = []
        for row in range(len(query_ids)):
            extras = copy_scores[row, :t]
            extra_ids = np.asarray(copy_ids[:t])
            doc_ids_arr = np.asarray(small_ids)
            scores = base_scores[row]
            for slot, doc_index in enumerate(gold_index[row].tolist()):
                gs = scores[doc_index]
                gid = doc_ids_arr[doc_index]
                better = int((scores > gs).sum() + (extras > gs).sum())
                tie = int(((scores == gs) & (doc_ids_arr < gid)).sum() + ((extras == gs) & (extra_ids < gid)).sum())
                ranks[row, slot] = better + tie + 1
            gold_scores = scores[gold_index[row]]
            gaps.append(float(gold_scores.min() - extras.max()))
        planted[f"t={t}"] = pack_metrics(ranks)
        planted[f"t={t}"]["median-gap-worse-gold-minus-best-copy"] = round(float(np.median(gaps)), 4)
        planted[f"t={t}"]["mean-gap-worse-gold-minus-best-copy"] = round(float(np.mean(gaps)), 4)
        planted[f"t={t}"]["copies-above-worse-gold"] = int(sum(g < 0 for g in gaps))
        planted[f"t={t}"]["copies-within-0.01-of-worse-gold"] = int(sum(abs(g) <= 0.01 for g in gaps))
    # Global index: 2 copies of every attribute sit in one corpus.
    global_copy = copy_emb[:, :2, :].reshape(len(query_ids) * 2, -1)
    global_scores_copies = q_orig_unit @ global_copy.T
    global_scores = np.concatenate([base_scores, global_scores_copies], axis=1)
    global_ids = small_ids + [f"Copy {row:04d}-{c}" for row in range(len(query_ids)) for c in range(2)]
    planted["global-t=2"] = metrics_from_scores(global_scores, gold_index, id_key(global_ids))
    print("planted", json.dumps(planted), flush=True)

    # --- late interaction ---
    print("token embeddings", flush=True)
    q_vecs, q_owner = token_packs(model, query_texts, drop_query_stops=False)
    q_vecs_c, q_owner_c = token_packs(model, query_texts, drop_query_stops=True)
    d_vecs, d_owner = token_packs(model, small_texts, drop_query_stops=False)
    maxsim_sum = maxsim_scores(q_vecs, q_owner, d_vecs, d_owner, len(query_ids), len(small_ids), "sum")
    maxsim_content = maxsim_scores(q_vecs_c, q_owner_c, d_vecs, d_owner, len(query_ids), len(small_ids), "sum")
    maxsim_best = maxsim_scores(q_vecs_c, q_owner_c, d_vecs, d_owner, len(query_ids), len(small_ids), "max")
    # Gloss queries, content MaxSim, original docs.
    g_vecs, g_owner = token_packs(model, gloss_query_texts, drop_query_stops=True)
    maxsim_gloss = maxsim_scores(g_vecs, g_owner, d_vecs, d_owner, len(gloss_index), len(small_ids), "sum")

    rng_aniso = np.random.default_rng(SEED)
    take_n = min(4000, d_vecs.shape[0])
    left = rng_aniso.integers(0, d_vecs.shape[0], take_n)
    right = rng_aniso.integers(0, d_vecs.shape[0], take_n)
    aniso = float(np.mean(np.sum(d_vecs[left] * d_vecs[right], axis=1)))

    # Late chunking: pool contextual tokens inside each attribute span with the
    # model's own pooling layer, then max over spans. Query side is the sentence vector.
    print("late chunking", flush=True)
    tokenizer = model.tokenizer
    doc_token_embs = [
        as_numpy(item) for item in model.encode(
            small_texts, batch_size=16, output_value="token_embeddings",
            normalize_embeddings=False, show_progress_bar=False, convert_to_numpy=True,
        )
    ]
    late_scores = np.full((len(query_ids), len(small_ids)), -1e9, dtype=np.float32)
    span_token_counts = []
    for doc_index, text in enumerate(small_texts):
        emb = np.asarray(doc_token_embs[doc_index], dtype=np.float32)
        encoded = tokenizer(text, add_special_tokens=True, truncation=True, max_length=256,
                            return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        if len(offsets) != len(emb):
            width = min(len(offsets), len(emb))
            offsets = offsets[:width]
            emb = emb[:width]
        spans = attr_char_spans(text, parsed[doc_index][1])
        span_idxs = []
        for start, end in spans:
            idxs = [i for i, (s, e) in enumerate(offsets)
                    if not (s == 0 and e == 0) and e > start and s < end]
            if not idxs:
                idxs = [i for i, (s, e) in enumerate(offsets) if not (s == 0 and e == 0)][:1]
            span_idxs.append(idxs)
            span_token_counts.append(len(idxs))
        n_spans = len(span_idxs)
        tensor = torch.tensor(emb, device=device).unsqueeze(0).repeat(n_spans, 1, 1)
        mask = torch.zeros(n_spans, emb.shape[0], device=device)
        for span_i, idxs in enumerate(span_idxs):
            mask[span_i, idxs] = 1
        with torch.no_grad():
            pooled = model[1]({"token_embeddings": tensor, "attention_mask": mask})
            vecs = pooled["sentence_embedding"].detach().float().cpu().numpy()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-8)
        late_scores[:, doc_index] = (q_bank["original"] @ vecs.T).max(axis=1)
    late_metrics = metrics_from_scores(late_scores, gold_index, small_key)
    # Same spans, but the query is the gloss subset. Re-pool is already in late_scores
    # for original queries; gloss queries need another max against the same span vectors.
    # Recompute gloss rows only by storing span vectors would be heavy. Encode is done
    # per doc above and discarded. Repeat the loop for gloss queries only if cheap:
    # 46 docs, instant relative to drowning. Do it.
    late_gloss = np.full((len(gloss_index), len(small_ids)), -1e9, dtype=np.float32)
    for doc_index, text in enumerate(small_texts):
        emb = np.asarray(doc_token_embs[doc_index], dtype=np.float32)
        encoded = tokenizer(text, add_special_tokens=True, truncation=True, max_length=256,
                            return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        if len(offsets) != len(emb):
            width = min(len(offsets), len(emb))
            offsets = offsets[:width]
            emb = emb[:width]
        spans = attr_char_spans(text, parsed[doc_index][1])
        span_idxs = []
        for start, end in spans:
            idxs = [i for i, (s, e) in enumerate(offsets)
                    if not (s == 0 and e == 0) and e > start and s < end]
            if not idxs:
                idxs = [1]
            span_idxs.append(idxs)
        n_spans = len(span_idxs)
        tensor = torch.tensor(emb, device=device).unsqueeze(0).repeat(n_spans, 1, 1)
        mask = torch.zeros(n_spans, emb.shape[0], device=device)
        for span_i, idxs in enumerate(span_idxs):
            mask[span_i, idxs] = 1
        with torch.no_grad():
            pooled = model[1]({"token_embeddings": tensor, "attention_mask": mask})
            vecs = pooled["sentence_embedding"].detach().float().cpu().numpy()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.maximum(norms, 1e-8)
        late_gloss[:, doc_index] = (q_bank["gloss-who-likes"] @ vecs.T).max(axis=1)

    late_interaction = {
        "pooling": pooling_name,
        "pool-module-agreement": round(pool_agreement, 4),
        "token-cosine-mean-random-pairs": round(aniso, 4),
        "mean-span-tokens": round(float(np.mean(span_token_counts)), 2),
        "maxsim-sum-all-tokens": metrics_from_scores(maxsim_sum, gold_index, small_key),
        "maxsim-sum-content-tokens": metrics_from_scores(maxsim_content, gold_index, small_key),
        "maxsim-best-content-token": metrics_from_scores(maxsim_best, gold_index, small_key),
        "maxsim-content-on-gloss-queries": metrics_from_scores(
            maxsim_gloss, gold_index[gloss_index], small_key
        ),
        "late-chunk-max": late_metrics,
        "late-chunk-max-on-gloss-queries": metrics_from_scores(
            late_gloss, gold_index[gloss_index], small_key
        ),
    }
    print("late", {k: (v.get("recall@2") if isinstance(v, dict) else v) for k, v in late_interaction.items()}, flush=True)

    # --- fusion ---
    whole_scores = q_bank["original"] @ d_small.T
    fusion = {
        "exact-bm25-whole": fusion_block(
            bm25_scores["original"], whole_scores, gold_index, small_key, "bm25", "whole-doc"
        ),
        "exact-bm25-chunkmax": fusion_block(
            bm25_scores["original"], chunk_orig_scores, gold_index, small_key, "bm25", "chunk-max"
        ),
        "exact-bm25-maxsim": fusion_block(
            bm25_scores["original"], maxsim_content, gold_index, small_key, "bm25", "maxsim-content"
        ),
        "exact-whole-chunkmax": fusion_block(
            whole_scores, chunk_orig_scores, gold_index, small_key, "whole-doc", "chunk-max"
        ),
        "drop-bm25-whole": fusion_block(
            bm25_scores["drop-attribute"], q_bank["drop-attribute"] @ d_small.T,
            gold_index, small_key, "bm25", "whole-doc",
        ),
        "drop-bm25-chunkmax": fusion_block(
            bm25_scores["drop-attribute"], chunk_scores("drop-attribute", "k1", "max"),
            gold_index, small_key, "bm25", "chunk-max",
        ),
    }
    # Gloss subset fusion. Matrices are subset-length.
    fusion["gloss-query-bm25-whole"] = fusion_block(
        bm25_gloss_q, dense_gloss_q, gold_index[gloss_index], small_key, "bm25", "whole-doc"
    )
    fusion["gloss-query-bm25-chunkmax"] = fusion_block(
        bm25_gloss_q, chunk_gloss_q, gold_index[gloss_index], small_key, "bm25", "chunk-max"
    )
    fusion["gloss-query-bm25-maxsim"] = fusion_block(
        bm25_gloss_q, maxsim_gloss, gold_index[gloss_index], small_key, "bm25", "maxsim-content"
    )
    fusion["docs-gloss-bm25-chunkmax"] = fusion_block(
        bm25_orig_on_rewritten[gloss_index], chunk_orig_on_rewritten[gloss_index],
        gold_index[gloss_index], small_key, "bm25", "chunk-max",
    )
    fusion["both-gloss-bm25-chunkmax"] = fusion_block(
        bm25_both, chunk_both, gold_index[gloss_index], small_key, "bm25", "chunk-max"
    )
    if len(strict_index):
        strict_pos = np.asarray([gloss_rows.index(int(r)) for r in strict_index], dtype=np.int32)
        fusion["strict-gloss-query-bm25-chunkmax"] = fusion_block(
            bm25_gloss_q[strict_pos], chunk_gloss_q[strict_pos],
            gold_index[strict_index], small_key, "bm25", "chunk-max",
        )
        fusion["strict-docs-gloss-bm25-chunkmax"] = fusion_block(
            bm25_orig_on_rewritten[strict_index], chunk_orig_on_rewritten[strict_index],
            gold_index[strict_index], small_key, "bm25", "chunk-max",
        )

    # Nonsense control: 1000 docs that do not share the attribute vocabulary.
    nonsense_texts = [f"Nobody Special likes qxzz{i} plugh." for i in range(1000)]
    nonsense_emb = encode_sentences(model, nonsense_texts, batch_size=256)
    nonsense_scores = np.concatenate([chunk_orig_scores, q_bank["original"] @ nonsense_emb.T], axis=1)
    nonsense_ids = small_ids + [f"Nonsense {i}" for i in range(1000)]
    nonsense_metrics = metrics_from_scores(nonsense_scores, gold_index, id_key(nonsense_ids))

    results = {
        "model": MODEL_NAME,
        "device": device,
        "pooling": pooling_name,
        "pool-module-agreement": round(pool_agreement, 4),
        "n-docs": len(small_ids),
        "n-queries": len(query_ids),
        "n-glosses": int(len(gloss_index)),
        "gloss-list-note": (
            "20 case-sensitive superstring collisions, seed-0 sample of 100 other "
            "attributes, plus Chairs, Limes, Barley, Joshua Trees, Disco Music, Soy Sauce"
        ),
        "reproduction": reproduction,
        "query-templates": query_template_metrics,
        "gloss": {
            "n": int(len(gloss_index)),
            "strict-lexical-cutoff-n": int(len(strict_index)),
            "median-min-residual-df": round(float(np.median(residual_df)), 2) if residual_df else None,
            "settings": gloss_settings,
            "cosine-margins-mean": margin_means,
            "cosine-margins-lowest-gloss-chunk": sorted(
                margins, key=lambda item: item["cos-orig-query-gloss-chunk"]
            )[:8],
            "cosine-margins-highest-gloss-chunk": sorted(
                margins, key=lambda item: item["cos-orig-query-gloss-chunk"]
            )[-8:],
        },
        "chunk-geometry": geometry,
        "collisions": {
            "n-queries": int(len(collision_index)),
            "chunk-max": metrics_from_scores(chunk_orig_scores[collision_index], gold_index[collision_index], small_key) if len(collision_index) else {},
            "bm25": metrics_from_scores(bm25_scores["original"][collision_index], gold_index[collision_index], small_key) if len(collision_index) else {},
            "whole-doc": metrics_from_scores(whole_scores[collision_index], gold_index[collision_index], small_key) if len(collision_index) else {},
            "detail": collision_detail,
        },
        "chunk-max-failures": {
            "n-queries-not-both-in-top2": len(fail_examples),
            "n-top-chunk-contains-attribute-string": sum(1 for ex in fail_examples if ex["attribute-in-best-chunk"]),
            "examples": fail_examples[:40],
        },
        "semantic-neighbors": semantic_neighbors,
        "planted-exact-copies": planted,
        "nonsense-1000": nonsense_metrics,
        "late-interaction": late_interaction,
        "fusion": fusion,
        "seconds-before-drowning": round(time.time() - t0, 1),
    }
    (HERE / "results.json").write_text(json.dumps(jsonable(results), indent=2))
    print(f"core saved in {results['seconds-before-drowning']}s", flush=True)

    # --- drowning ---
    print("loading full corpus", flush=True)
    full_docs = load_jsonl(FULL / "corpus.jsonl")
    full_ids = [row["_id"] for row in full_docs]
    if set(full_ids) & set(small_ids):
        raise SystemExit("expected zero id overlap between draws")
    full_texts = [doc_text(row) for row in full_docs]
    perm = np.random.default_rng(SEED).permutation(len(full_docs))
    query_automaton = build_aho(query_attrs)

    print("scanning distractors for attribute strings", flush=True)
    t_scan = time.time()
    # contains_full[q, i] over file order, then reorder.
    contains_full = np.zeros((len(query_ids), len(full_docs)), dtype=bool)
    full_parsed: list[tuple[str, list[str]] | None] = [None] * len(full_docs)
    n_parse_fail = 0
    for doc_index, text in enumerate(full_texts):
        hits = aho_find(text, query_automaton)
        if hits:
            contains_full[hits, doc_index] = True
        try:
            full_parsed[doc_index] = split_likes(text)
        except ValueError:
            n_parse_fail += 1
            full_parsed[doc_index] = (full_ids[doc_index], [text])
        if doc_index and doc_index % 10000 == 0:
            print(f"  scanned {doc_index}", flush=True)
    print(f"scan {round(time.time() - t_scan, 1)}s parse-fails {n_parse_fail}", flush=True)

    # Cache Aho hits on unique strings used in chunk text: names and attributes.
    hit_cache: dict[str, np.ndarray] = {}

    def hits_of(text: str) -> np.ndarray:
        cached = hit_cache.get(text)
        if cached is None:
            cached = np.asarray(aho_find(text, query_automaton), dtype=np.int32)
            hit_cache[text] = cached
        return cached

    print("embedding full documents for whole-doc curve", flush=True)
    d_full = encode_sentences(model, full_texts, batch_size=512)
    # Order: 46 small + perm.
    added_ids = [full_ids[i] for i in perm]
    curve_ids = small_ids + added_ids
    curve_key = id_key(curve_ids)
    d_curve = np.concatenate([d_small, d_full[perm]], axis=0)
    whole_curve_scores = q_bank["original"] @ d_curve.T
    contains_added = contains_full[:, perm]  # (nq, 50000)

    def mask_variants(scores: np.ndarray, n_added: int) -> dict[str, np.ndarray]:
        """scores columns are small + perm[:all]. Use only the first 46+n_added."""
        n = 46 + n_added
        full = scores[:, :n]
        sem = full.copy()
        exact = np.full_like(full, -1e9)
        exact[:, :46] = full[:, :46]
        if n_added:
            hit = contains_added[:, :n_added]
            sem[:, 46:] = np.where(hit, -1e9, full[:, 46:])
            exact[:, 46:] = np.where(hit, full[:, 46:], -1e9)
        return {"full": full, "sem": sem, "exact": exact}

    # Chunk-MaxP scores, preallocated. Gold chunks first.
    n_curve = 46 + len(perm)
    maxp_full = np.full((len(query_ids), n_curve), -1e9, dtype=np.float32)
    maxp_sem = np.full((len(query_ids), n_curve), -1e9, dtype=np.float32)
    maxp_exact = np.full((len(query_ids), n_curve), -1e9, dtype=np.float32)
    gold_chunk_scores = chunk_orig_scores  # (nq, 46)
    maxp_full[:, :46] = gold_chunk_scores
    maxp_sem[:, :46] = gold_chunk_scores
    maxp_exact[:, :46] = gold_chunk_scores

    print("embedding distractor chunks", flush=True)
    t_chunks = time.time()
    batch_docs = 64
    n_added_total = len(perm)
    for start in range(0, n_added_total, batch_docs):
        end = min(start + batch_docs, n_added_total)
        texts_batch: list[str] = []
        owners_local: list[int] = []
        # Per chunk, query indices hit by the attribute string.
        hit_lists: list[np.ndarray] = []
        for offset, src in enumerate(perm[start:end]):
            name, attrs = full_parsed[int(src)]
            name_hits = hits_of(name)
            doc_slot = 46 + start + offset
            for attr in attrs:
                texts_batch.append(f"{name} likes {attr}.")
                owners_local.append(doc_slot)
                attr_hits = hits_of(attr)
                if len(name_hits) and len(attr_hits):
                    hit_lists.append(np.union1d(name_hits, attr_hits))
                elif len(name_hits):
                    hit_lists.append(name_hits)
                else:
                    hit_lists.append(attr_hits)
        if not texts_batch:
            continue
        emb = encode_sentences(model, texts_batch, batch_size=512)
        sims = q_bank["original"] @ emb.T  # (nq, n_chunks)
        owners_arr = np.asarray(owners_local, dtype=np.int32)
        for doc_slot in range(46 + start, 46 + end):
            cols = np.flatnonzero(owners_arr == doc_slot)
            if cols.size == 0:
                continue
            block = sims[:, cols]
            maxp_full[:, doc_slot] = block.max(axis=1)
            hit = np.zeros(block.shape, dtype=bool)
            for local_col, src_col in enumerate(cols.tolist()):
                idxs = hit_lists[src_col]
                if len(idxs):
                    hit[idxs, local_col] = True
            sem_block = np.where(hit, -1e9, block)
            exact_block = np.where(hit, block, -1e9)
            maxp_sem[:, doc_slot] = sem_block.max(axis=1)
            maxp_exact[:, doc_slot] = exact_block.max(axis=1)
        if end % 2048 == 0 or end == n_added_total:
            print(f"  distractor docs {end} chunks-in-batch {len(texts_batch)} "
                  f"elapsed {round(time.time() - t_chunks, 1)}s", flush=True)

    # BM25 on small + prefix. Tokenize once in curve order.
    print("bm25 drowning", flush=True)
    curve_texts = small_texts + [full_texts[int(i)] for i in perm]
    curve_tokens = [tokenize(text) for text in curve_texts]
    # Postings over the full curve; prefix df via searchsorted.
    postings: dict[str, list[int]] = {}
    tfs: dict[str, list[float]] = {}
    doc_len = np.asarray([len(tokens) for tokens in curve_tokens], dtype=np.float64)
    for doc_index, tokens in enumerate(curve_tokens):
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        for token, tf in counts.items():
            postings.setdefault(token, []).append(doc_index)
            tfs.setdefault(token, []).append(float(tf))
    post_idx = {token: np.asarray(idx, dtype=np.int32) for token, idx in postings.items()}
    post_tf = {token: np.asarray(tf, dtype=np.float64) for token, tf in tfs.items()}
    q_token_lists = [list(dict.fromkeys(tokenize(text))) for text in query_texts]

    def bm25_prefix(n_active: int) -> np.ndarray:
        scores = np.zeros((len(query_ids), n_active), dtype=np.float32)
        avgdl = float(doc_len[:n_active].mean())
        lengths = doc_len[:n_active]
        for row, tokens in enumerate(q_token_lists):
            for token in tokens:
                idx_all = post_idx.get(token)
                if idx_all is None:
                    continue
                cut = int(np.searchsorted(idx_all, n_active))
                if cut == 0:
                    continue
                idx = idx_all[:cut]
                tf = post_tf[token][:cut]
                idf = lucene_idf(n_active, cut)
                norm = 1.0 - B + B * lengths[idx] / avgdl
                denom = tf + K1 * norm
                scores[row, idx] += np.float32(idf * (tf * (K1 + 1.0)) / denom)
        return scores

    curve = []
    for n_added in ADDED:
        if n_added > n_added_total:
            break
        n_active = 46 + n_added
        key = id_key(curve_ids[:n_active])
        whole_vars = mask_variants(whole_curve_scores, n_added)
        # Chunk masks were applied per chunk. Doc-level string mask is a
        # different object; both are reported. Chunk matrices already cover
        # the prefix because later columns stay -1e9... NO: maxp_full has ALL
        # distractors filled. Slice to the prefix.
        def sliced(mat: np.ndarray) -> np.ndarray:
            return mat[:, :n_active]

        row = {
            "added": int(n_added),
            "total": int(n_active),
            "whole-doc": metrics_from_scores(whole_vars["full"], gold_index, key),
            "whole-semantic-only": metrics_from_scores(whole_vars["sem"], gold_index, key),
            "whole-exact-string-only": metrics_from_scores(whole_vars["exact"], gold_index, key),
            "chunk-max": metrics_from_scores(sliced(maxp_full), gold_index, key),
            "chunk-max-semantic-only": metrics_from_scores(sliced(maxp_sem), gold_index, key),
            "chunk-max-exact-string-only": metrics_from_scores(sliced(maxp_exact), gold_index, key),
            "bm25": metrics_from_scores(bm25_prefix(n_active), gold_index, key),
        }
        if n_added:
            counts = contains_added[:, :n_added].sum(axis=1)
        else:
            counts = np.zeros(len(query_ids), dtype=np.int32)
        row["exact-string-distractors"] = {
            "median": round(float(np.median(counts)), 2),
            "mean": round(float(counts.mean()), 2),
            "queries-with-zero": int((counts == 0).sum()),
            "queries-with-at-least-one": int((counts > 0).sum()),
        }
        bins = []
        edges = [(0, 0), (1, 2), (3, 10), (11, 100), (101, 10**9)]
        chunk_ranks_now = gold_ranks(sliced(maxp_full), gold_index, key)
        for lo, hi in edges:
            mask = (counts >= lo) & (counts <= hi)
            label = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**8 else f"{lo}+")
            if not np.any(mask):
                bins.append({"bin": label, "n": 0})
                continue
            bins.append({
                "bin": label,
                "n": int(mask.sum()),
                "chunk-max": pack_metrics(chunk_ranks_now[mask]),
            })
        row["chunk-max-by-exact-distractors"] = bins
        curve.append(row)
        print(
            f"added {n_added}: whole {row['whole-doc']['recall@2']} "
            f"maxp {row['chunk-max']['recall@2']} "
            f"maxp-sem {row['chunk-max-semantic-only']['recall@2']} "
            f"maxp-exact {row['chunk-max-exact-string-only']['recall@2']} "
            f"bm25 {row['bm25']['recall@2']} "
            f"zero-string-queries {row['exact-string-distractors']['queries-with-zero']}",
            flush=True,
        )

    results["drowning"] = {
        "added-are": "seed-0 permutation of data/limit, a different draw from limit-small",
        "semantic-only": "distractor chunks/docs whose text contains the query attribute string are removed",
        "exact-string-only": "distractor chunks/docs that do not contain the attribute string are removed",
        "parse-fails": n_parse_fail,
        "curve": curve,
        "nonsense-1000-chunk-max": nonsense_metrics,
    }
    results["seconds"] = round(time.time() - t0, 1)
    (HERE / "results.json").write_text(json.dumps(jsonable(results), indent=2))
    write_report(results, HERE / "report.md")
    print(f"done in {results['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
