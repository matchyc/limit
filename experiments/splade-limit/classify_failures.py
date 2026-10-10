#!/usr/bin/env python3
"""Classify SPLADE false positives on LIMIT-50k.

Question: do SPLADE failures all come from WordPiece / name splitting
(Arta → art), or also from likes-item superstrings and MLM expansion?

Run:
  CUDA_VISIBLE_DEVICES=1 ../zero-shot-embed/.venv/bin/python classify_failures.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
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
WORD_RE = re.compile(r"[a-z0-9]+")

sys.path.insert(0, str(SCHEME))
from run_scheme_failures import (  # noqa: E402
    attribute_of,
    doc_text,
    gold_ranks,
    id_key,
    load_jsonl,
    split_likes,
)

STOP = {
    "a", "an", "the", "of", "and", "or", "to", "for", "in", "on", "with",
    "from", "who", "likes",
}


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "to_dense"):
        value = value.to_dense()
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def content_words(text: str) -> list[str]:
    tokens = words(text)
    kept = [t for t in tokens if t not in STOP]
    return kept or tokens


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


def shared_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def name_collision(name_tokens: list[str], name_wp: list[str], q_words: list[str]) -> str:
    """Return a short reason if the person name collides with the query attribute."""
    q_set = set(q_words)
    for ntok in name_tokens:
        for w in q_words:
            if len(w) >= 2 and ntok.startswith(w):
                return f"name_prefix:{w}|{ntok}"
            if len(ntok) >= 3 and w.startswith(ntok):
                return f"name_is_prefix:{ntok}|{w}"
            if shared_prefix(ntok, w) >= 4:
                return f"name_stem:{ntok[:4]}|{ntok}|{w}"
        for piece in name_wp:
            surf = piece[2:] if piece.startswith("##") else piece
            if len(surf) >= 2 and surf in q_set:
                return f"name_wp:{piece}|{ntok}"
    return ""


def likes_flags(attr_words: list[str], q_words: list[str], likes: list[str]) -> tuple[str, str, str]:
    """Return (phrase, word, affix) reasons against likes-items."""
    phrase = ""
    word_hit = ""
    affix = ""
    q_set = set(q_words)
    for item in likes:
        item_words = words(item)
        if not item_words:
            continue
        exact = item_words == attr_words
        if (not exact) and is_subseq(attr_words, item_words) and not phrase:
            phrase = item
        if exact:
            continue
        for iw in item_words:
            if iw in q_set and not word_hit:
                word_hit = f"{iw}|{item}"
            for w in q_words:
                if iw == w:
                    continue
                if len(w) >= 2 and (iw.startswith(w) or iw.endswith(w)):
                    if not affix:
                        affix = f"{w}|{iw}|{item}"
                elif len(w) >= 4 and shared_prefix(iw, w) >= 4 and not affix:
                    affix = f"stem:{w}|{iw}|{item}"
    return phrase, word_hit, affix


def is_subseq(needle: list[str], hay: list[str]) -> bool:
    n, h = len(needle), len(hay)
    if n == 0 or n > h:
        return False
    for i in range(h - n + 1):
        if hay[i : i + n] == needle:
            return True
    return False


def primary_label(name_r: str, phrase: str, word_hit: str, affix: str) -> str:
    if name_r:
        return "name_split"
    if phrase:
        return "attr_phrase"
    if word_hit:
        return "attr_word"
    if affix:
        return "attr_affix"
    return "expansion"


def wp_name(tokenizer, name: str) -> list[str]:
    return [t.lower() for t in tokenizer.tokenize(name)]


def encode_scores(model, query_texts: list[str], doc_texts: list[str]) -> tuple[np.ndarray, object]:
    n_docs = len(doc_texts)
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
        if start == 0 or end == n_docs or end % 3200 == 0:
            print(f"encoded docs {end}/{n_docs}", flush=True)
        del d_emb
        torch.cuda.empty_cache()
    return scores, q_emb


def mask_eval(scores: np.ndarray, gold_index: np.ndarray, key: np.ndarray, drop: np.ndarray) -> dict:
    """drop is (n_queries, n_docs) bool. Golds are never dropped."""
    masked = scores.copy()
    masked[drop] = np.float32(-1.0e9)
    for row, golds in enumerate(gold_index):
        for g in golds:
            masked[row, int(g)] = scores[row, int(g)]
    return pack(gold_ranks(masked, gold_index, key))


def pick_examples(rows: list[dict], label: str, n: int = 4) -> list[dict]:
    out = []
    for row in rows:
        if row["top1-label"] != label:
            continue
        out.append({
            "query": row["query"],
            "attribute": row["attribute"],
            "splade-gold-ranks": row["splade-gold-ranks"],
            "top1": row["top1"],
            "top1-reason": row["top1-reason"],
            "top10-fp-labels": row["top10-fp-labels"],
            "query-expansion": row["query-expansion"][:8],
            "top1-expansion-overlap": row["top1-expansion-overlap"],
        })
        if len(out) >= n:
            break
    return out


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
    parsed = [split_likes(text) for text in doc_texts]
    names = [name for name, _attrs in parsed]
    likes = [attrs for _name, attrs in parsed]
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
    print(f"docs={n_docs} queries={n_queries}", flush=True)

    from sentence_transformers import SparseEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {MODEL_NAME} on {device}", flush=True)
    model = SparseEncoder(MODEL_NAME, device=device)
    tokenizer = model.tokenizer

    scores, q_emb = encode_scores(model, query_texts, doc_texts)
    ranks = gold_ranks(scores, gold_index, key)
    base = pack(ranks)
    print("splade", base, flush=True)

    q_decoded = model.decode(q_emb, top_k=24)
    attr_word_sets = [set(content_words(attr)) for attr in query_attrs]
    q_exp_terms = []
    for decoded, attr_set in zip(q_decoded, attr_word_sets):
        extra = []
        for tok, weight in decoded:
            surf = tok[2:] if tok.startswith("##") else tok
            surf = surf.lower()
            if surf in STOP or surf in attr_set:
                continue
            extra.append((tok, round(float(weight), 3)))
        q_exp_terms.append(extra)

    name_tokens = [words(name) for name in names]
    name_wp = [wp_name(tokenizer, name) for name in names]
    q_words_list = [content_words(attr) for attr in query_attrs]
    attr_words_list = [words(attr) for attr in query_attrs]

    name_drop = np.zeros((n_queries, n_docs), dtype=bool)
    phrase_drop = np.zeros((n_queries, n_docs), dtype=bool)
    word_drop = np.zeros((n_queries, n_docs), dtype=bool)
    affix_drop = np.zeros((n_queries, n_docs), dtype=bool)

    token_docs: dict[str, list[int]] = defaultdict(list)
    wp_docs: dict[str, list[int]] = defaultdict(list)
    for di, (ntoks, nwp) in enumerate(zip(name_tokens, name_wp)):
        for tok in set(ntoks):
            token_docs[tok].append(di)
        for piece in nwp:
            surf = piece[2:] if piece.startswith("##") else piece
            if len(surf) >= 2:
                wp_docs[surf].append(di)

    item_to_docs: dict[str, list[int]] = defaultdict(list)
    item_words: dict[str, list[str]] = {}
    for di, items in enumerate(likes):
        for item in items:
            item_to_docs[item].append(di)
            if item not in item_words:
                item_words[item] = words(item)
    print(
        f"unique likes-items={len(item_words)} unique name-tokens={len(token_docs)}",
        flush=True,
    )

    for ntok, dis in token_docs.items():
        dis_arr = np.asarray(dis, dtype=np.int32)
        for qi, q_w in enumerate(q_words_list):
            hit = False
            for w in q_w:
                if len(w) >= 2 and ntok.startswith(w):
                    hit = True
                    break
                if len(ntok) >= 3 and w.startswith(ntok):
                    hit = True
                    break
                if shared_prefix(ntok, w) >= 4:
                    hit = True
                    break
            if hit:
                name_drop[qi, dis_arr] = True
    for qi, q_w in enumerate(q_words_list):
        for w in q_w:
            if w in wp_docs:
                name_drop[qi, wp_docs[w]] = True

    for qi, (attr_w, q_w) in enumerate(zip(attr_words_list, q_words_list)):
        q_set = set(q_w)
        for item, iw in item_words.items():
            if iw == attr_w:
                continue
            docs_i = item_to_docs[item]
            if is_subseq(attr_w, iw):
                phrase_drop[qi, docs_i] = True
            if any(tok in q_set for tok in iw):
                word_drop[qi, docs_i] = True
            hit_affix = False
            for tok in iw:
                if hit_affix:
                    break
                for w in q_w:
                    if tok == w:
                        continue
                    if len(w) >= 2 and (tok.startswith(w) or tok.endswith(w)):
                        hit_affix = True
                        break
                    if len(w) >= 4 and shared_prefix(tok, w) >= 4:
                        hit_affix = True
                        break
            if hit_affix:
                affix_drop[qi, docs_i] = True
        if (qi + 1) % 200 == 0 or qi + 1 == n_queries:
            print(f"likes-masks {qi + 1}/{n_queries}", flush=True)

    string_drop = name_drop | phrase_drop | word_drop | affix_drop
    ablations = {
        "unfiltered": base,
        "drop-name-split": mask_eval(scores, gold_index, key, name_drop),
        "drop-attr-phrase": mask_eval(scores, gold_index, key, phrase_drop),
        "drop-attr-word": mask_eval(scores, gold_index, key, word_drop),
        "drop-attr-affix": mask_eval(scores, gold_index, key, affix_drop),
        "drop-name-and-phrase": mask_eval(scores, gold_index, key, name_drop | phrase_drop),
        "drop-all-string-collisions": mask_eval(scores, gold_index, key, string_drop),
    }
    for name, metrics in ablations.items():
        print(name, metrics, flush=True)

    failed = ~((ranks <= 2).all(axis=1))
    per_query = []
    top1_labels = Counter()
    failed_top1 = Counter()
    failed_majority = Counter()
    top10_fp_labels = Counter()
    failed_any = Counter()

    for qi in range(n_queries):
        order = np.lexsort((key, -scores[qi]))
        golds = set(int(g) for g in gold_index[qi])
        attr = query_attrs[qi]
        q_w = q_words_list[qi]
        attr_w = attr_words_list[qi]
        top10 = []
        fp_labels = []
        for rank, di in enumerate(order[:10], start=1):
            di = int(di)
            name_r = name_collision(name_tokens[di], name_wp[di], q_w)
            phrase, word_hit, affix = likes_flags(attr_w, q_w, likes[di])
            label = "gold" if di in golds else primary_label(name_r, phrase, word_hit, affix)
            rec = {
                "rank": rank,
                "id": doc_ids[di],
                "score": round(float(scores[qi, di]), 4),
                "is-gold": di in golds,
                "label": label,
                "name-reason": name_r,
                "attr-phrase": phrase,
                "attr-word": word_hit,
                "attr-affix": affix,
            }
            top10.append(rec)
            if di not in golds:
                fp_labels.append(label)
                top10_fp_labels[label] += 1
        top1 = top10[0]
        top1_labels[top1["label"]] += 1
        flags = {
            "name_split": bool(name_drop[qi, int(order[0])] and int(order[0]) not in golds)
            or top1["label"] == "name_split",
            "attr_phrase": top1["label"] == "attr_phrase",
            "attr_word": top1["label"] == "attr_word",
            "attr_affix": top1["label"] == "attr_affix",
            "expansion": top1["label"] == "expansion",
            "gold": top1["label"] == "gold",
        }
        any_fp = {
            "name_split": any(x["label"] == "name_split" for x in top10 if not x["is-gold"]),
            "attr_phrase": any(x["label"] == "attr_phrase" for x in top10 if not x["is-gold"]),
            "attr_word": any(x["label"] == "attr_word" for x in top10 if not x["is-gold"]),
            "attr_affix": any(x["label"] == "attr_affix" for x in top10 if not x["is-gold"]),
            "expansion": any(x["label"] == "expansion" for x in top10 if not x["is-gold"]),
        }
        majority = Counter(fp_labels).most_common(1)[0][0] if fp_labels else "none"
        overlap = []
        if not top1["is-gold"]:
            blob = " ".join(words(names[int(order[0])]) + [w for item in likes[int(order[0])] for w in words(item)])
            blob_set = set(blob.split())
            for tok, _wt in q_exp_terms[qi]:
                surf = tok[2:] if tok.startswith("##") else tok
                if surf.lower() in blob_set:
                    overlap.append(tok)
        row = {
            "query-id": query_ids[qi],
            "query": query_texts[qi],
            "attribute": attr,
            "failed-both-in-top2": bool(failed[qi]),
            "splade-gold-ranks": ranks[qi].tolist(),
            "top1": {"id": top1["id"], "label": top1["label"], "score": top1["score"]},
            "top1-label": top1["label"],
            "top1-reason": {
                "name": top1["name-reason"],
                "phrase": top1["attr-phrase"],
                "word": top1["attr-word"],
                "affix": top1["attr-affix"],
            },
            "top10-fp-labels": dict(Counter(fp_labels)),
            "majority-fp-label": majority,
            "any-fp-in-top10": any_fp,
            "query-expansion": q_exp_terms[qi],
            "top1-expansion-overlap": overlap[:8],
            "corpus-name-split": int(name_drop[qi].sum()),
            "corpus-attr-phrase": int(phrase_drop[qi].sum()),
        }
        per_query.append(row)
        if failed[qi]:
            failed_top1[top1["label"]] += 1
            failed_majority[majority] += 1
            for k, v in any_fp.items():
                if v:
                    failed_any[k] += 1

    n_failed = int(failed.sum())
    # Enrichment: name-split share of corpus vs of top-10 FPs on failed queries.
    corpus_name_rate = float(name_drop.mean())
    failed_idx = np.flatnonzero(failed)
    top10_name = 0
    top10_fp = 0
    for qi in failed_idx.tolist():
        order = np.lexsort((key, -scores[qi]))
        golds = set(int(g) for g in gold_index[qi])
        for di in order[:10]:
            di = int(di)
            if di in golds:
                continue
            top10_fp += 1
            if name_drop[qi, di]:
                top10_name += 1

    summary = {
        "n-queries": n_queries,
        "n-failed-both-in-top2": n_failed,
        "splade": base,
        "ablations": ablations,
        "all-queries-top1-label": dict(top1_labels),
        "failed-top1-label": dict(failed_top1),
        "failed-majority-fp-label": dict(failed_majority),
        "failed-any-fp-in-top10": dict(failed_any),
        "top10-fp-label-counts": dict(top10_fp_labels),
        "name-split-corpus-rate": round(corpus_name_rate, 5),
        "failed-top10-fp-name-split-rate": round(top10_name / max(top10_fp, 1), 4),
        "examples": {
            "name_split": pick_examples(per_query, "name_split"),
            "attr_phrase": pick_examples(per_query, "attr_phrase"),
            "attr_word": pick_examples(per_query, "attr_word"),
            "attr_affix": pick_examples(per_query, "attr_affix"),
            "expansion": pick_examples([r for r in per_query if r["failed-both-in-top2"]], "expansion"),
        },
    }

    known = {
        "Who likes Art History?": "name_split",
        "Who likes Honey?": "attr_phrase",
        "Who likes Psychology?": "expansion",
        "Who likes Saxophones?": "name_split",
        "Who likes Go?": "name_split",
    }
    checks = {}
    q_to_row = {r["query"]: r for r in per_query}
    for q, expect in known.items():
        got = q_to_row[q]["top1-label"]
        checks[q] = {"expected": expect, "got": got, "ok": got == expect}
        print("check", q, checks[q], "top1", q_to_row[q]["top1"], flush=True)

    out = {
        "model": MODEL_NAME,
        "seconds": round(time.time() - t0, 1),
        "summary": summary,
        "sanity-checks": checks,
        "failed-queries": [r for r in per_query if r["failed-both-in-top2"]],
    }
    (HERE / "failure_taxonomy.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    write_report(out, HERE / "failure_taxonomy.md")
    print("wrote failure_taxonomy.json", flush=True)


def pct(n: int, d: int) -> str:
    if d == 0:
        return "0.0"
    return f"{100.0 * n / d:.1f}"


def write_report(out: dict, path: Path) -> None:
    s = out["summary"]
    n_failed = s["n-failed-both-in-top2"]
    lines = [
        "# SPLADE failure taxonomy on LIMIT-50k",
        "",
        f"Model `{out['model']}`. {out['seconds']}s. "
        "A query fails if SPLADE does not put both golds in the top 2.",
        "",
        f"Failed queries: {n_failed} / {s['n-queries']}.",
        "",
        "## Top-1 false-positive label on failed queries",
        "",
        "| label | n | % of failed |",
        "| --- | ---: | ---: |",
    ]
    for label in ("name_split", "attr_phrase", "attr_word", "attr_affix", "expansion", "gold"):
        n = s["failed-top1-label"].get(label, 0)
        lines.append(f"| {label} | {n} | {pct(n, n_failed)} |")
    lines += [
        "",
        "Labels: `name_split` = query tokens collide with the person name "
        "(Arta→art, Goebel→go, Snowden→snow). `attr_phrase` = a different likes-item "
        "contains the full query as words (Honey Bees). `attr_word` = a query word "
        "appears as a whole word in another likes-item (Holly Trees for Birch Trees). "
        "`attr_affix` = query word is a prefix/suffix/stem of a likes word "
        "(Seahorses, Historical Fiction). `expansion` = none of those; MLM neighbors.",
        "",
        "## Ablation: drop colliding documents, keep golds",
        "",
        "| filter | R@2 | R@10 | R@100 | both@2 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, m in s["ablations"].items():
        lines.append(
            f"| {name} | {m['recall@2']} | {m['recall@10']} | {m['recall@100']} | {m['both-in-top2']} |"
        )
    lines += [
        "",
        f"Name-split documents are {100 * s['name-split-corpus-rate']:.2f}% of the corpus "
        f"but {100 * s['failed-top10-fp-name-split-rate']:.1f}% of top-10 false positives "
        "on failed queries.",
        "",
        "## Examples",
        "",
    ]
    for label, rows in s["examples"].items():
        lines += [f"### {label}", ""]
        if not rows:
            lines += ["(none)", ""]
            continue
        for row in rows:
            lines.append(
                f"- `{row['query']}` top1 `{row['top1']['id']}` "
                f"golds {row['splade-gold-ranks']} reason {row['top1-reason']}"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
