#!/usr/bin/env python3
"""Lexical baselines for the LIMIT dataset (Weller et al., arXiv:2508.21038).

BM25 (k1=1.5, b=0.75) over lowercase whitespace/punctuation tokens, plus a
raw-attribute substring baseline. Writes results.json in this directory.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/mnt/raid0nvme0/yangshen/meng/vector_search/limit")
DATA_ROOT = ROOT / "data"
OUT_DIR = ROOT / "experiments" / "lexical-bm25"

K1 = 1.5
B = 0.75
TOKEN_RE = re.compile(r"[a-z0-9]+")
QUERY_PREFIX = "Who likes "

PAPER_RECALL = {
    "limit-small": {2: 97.8, 10: 100.0, 20: 100.0},
    "limit": {2: 85.7, 10: 90.4, 100: 93.6},
}
EVAL_KS = {
    "limit-small": (2, 10, 20),
    "limit": (2, 10, 100),
}


def tokenize(text: str) -> list[str]:
    """Lowercase and split on whitespace and punctuation.

    A token is a maximal run of ASCII letters or digits. Apostrophes, hyphens,
    colons, and other punctuation are separators, so "Rubik's" -> ["rubik", "s"]
    and "Cross-country" -> ["cross", "country"].
    """
    return TOKEN_RE.findall(text.lower())


def extract_attribute(query_text: str) -> str:
    if not query_text.startswith(QUERY_PREFIX) or not query_text.endswith("?"):
        raise ValueError(f"unexpected query format: {query_text!r}")
    return query_text[len(QUERY_PREFIX) : -1]


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def lucene_idf(n_docs: int, df: int) -> float:
    """Lucene / ATIRE non-negative IDF: ln(1 + (N - df + 0.5) / (df + 0.5))."""
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


def bm25_scores(
    query_tokens: list[str],
    postings_idx: dict[str, np.ndarray],
    postings_tf: dict[str, np.ndarray],
    idf: dict[str, float],
    doc_len: np.ndarray,
    avgdl: float,
) -> np.ndarray:
    """Document-side Okapi BM25 with Lucene IDF. Each query type is scored once."""
    scores = np.zeros(doc_len.shape[0], dtype=np.float64)
    seen: set[str] = set()
    for token in query_tokens:
        if token in seen or token not in idf:
            continue
        seen.add(token)
        idx = postings_idx[token]
        tf = postings_tf[token]
        norm = 1.0 - B + B * doc_len[idx] / avgdl
        denom = tf + K1 * norm
        scores[idx] += idf[token] * (tf * (K1 + 1.0)) / denom
    return scores


def rank_order(scores: np.ndarray, id_rank: np.ndarray) -> np.ndarray:
    """Higher score first. Equal scores break toward the lexicographically smaller _id."""
    return np.lexsort((id_rank, -scores))


def docs_with_all_tokens(
    tokens: list[str], postings_idx: dict[str, np.ndarray]
) -> list[int]:
    """Doc indices whose token sets contain every token. Empty if any token is missing."""
    unique = list(dict.fromkeys(tokens))
    if not unique or any(token not in postings_idx for token in unique):
        return []
    unique.sort(key=lambda token: int(postings_idx[token].shape[0]))
    candidates = postings_idx[unique[0]]
    if len(unique) == 1:
        return [int(doc_id) for doc_id in candidates]

    def contains(token: str, doc_id: int) -> bool:
        posting = postings_idx[token]
        position = int(np.searchsorted(posting, doc_id))
        return position < posting.shape[0] and int(posting[position]) == doc_id

    matched: list[int] = []
    for doc_id_np in candidates:
        doc_id = int(doc_id_np)
        if all(contains(token, doc_id) for token in unique[1:]):
            matched.append(doc_id)
    return matched


def distribution(counts: list[int]) -> dict:
    values = np.asarray(counts, dtype=np.int64)
    histogram = {str(key): int(value) for key, value in sorted(Counter(counts).items())}
    return {
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "histogram": histogram,
    }


def build_index(texts: list[str]) -> dict:
    n_docs = len(texts)
    doc_len = np.zeros(n_docs, dtype=np.float64)
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for doc_id, text in enumerate(texts):
        counts = Counter(tokenize(text))
        doc_len[doc_id] = float(sum(counts.values()))
        for token, tf in counts.items():
            grouped[token].append((doc_id, tf))

    postings_idx: dict[str, np.ndarray] = {}
    postings_tf: dict[str, np.ndarray] = {}
    idf: dict[str, float] = {}
    for token, pairs in grouped.items():
        postings_idx[token] = np.asarray([doc_id for doc_id, _ in pairs], dtype=np.int32)
        postings_tf[token] = np.asarray([tf for _, tf in pairs], dtype=np.float64)
        idf[token] = lucene_idf(n_docs, len(pairs))

    return {
        "doc_len": doc_len,
        "avgdl": float(doc_len.mean()),
        "postings_idx": postings_idx,
        "postings_tf": postings_tf,
        "idf": idf,
        "vocab_size": len(idf),
        "n_docs": n_docs,
    }


def contains_token(postings_idx: dict[str, np.ndarray], token: str, doc_id: int) -> bool:
    posting = postings_idx.get(token)
    if posting is None:
        return False
    position = int(np.searchsorted(posting, doc_id))
    return position < posting.shape[0] and int(posting[position]) == doc_id


def evaluate_dataset(name: str) -> dict:
    data_dir = DATA_ROOT / name
    queries = load_jsonl(data_dir / "queries.jsonl")
    corpus = load_jsonl(data_dir / "corpus.jsonl")
    qrel_rows = load_jsonl(data_dir / "qrels.jsonl")

    doc_ids = [row["_id"] for row in corpus]
    if len(doc_ids) != len(set(doc_ids)):
        raise AssertionError(f"{name}: duplicate document ids")
    texts = [row["text"] for row in corpus]
    titles = [row.get("title", "") for row in corpus]
    id_to_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}

    alphabetical = np.argsort(np.asarray(doc_ids), kind="mergesort")
    id_rank = np.empty(len(doc_ids), dtype=np.int32)
    id_rank[alphabetical] = np.arange(len(doc_ids), dtype=np.int32)

    relevant: dict[str, list[str]] = defaultdict(list)
    for row in qrel_rows:
        if int(row["score"]) <= 0:
            continue
        relevant[row["query-id"]].append(row["corpus-id"])

    query_ids = [row["_id"] for row in queries]
    if len(query_ids) != len(set(query_ids)):
        raise AssertionError(f"{name}: duplicate query ids")
    for query_id in query_ids:
        rel = relevant[query_id]
        if len(rel) != 2 or len(set(rel)) != 2:
            raise AssertionError(f"{name}: {query_id} has relevant set {rel}")
        missing = [doc_id for doc_id in rel if doc_id not in id_to_index]
        if missing:
            raise AssertionError(f"{name}: {query_id} missing docs {missing}")

    print(f"[{name}] indexing {len(corpus)} documents", flush=True)
    index = build_index(texts)
    ks = EVAL_KS[name]

    bm25_sum = {k: 0.0 for k in ks}
    substr_sum = {k: 0.0 for k in ks}
    bm25_hit_hist = {k: Counter() for k in ks}
    substr_hit_hist = {k: Counter() for k in ks}

    both_relevant = 0
    any_irrelevant = 0
    irrelevant_counts: list[int] = []
    casefold_only_both = 0

    n_token_complete_equals_relevant = 0
    n_exact_equals_relevant = 0
    n_rarest_df_eq_2 = 0
    n_noncandidate_outranks = 0
    n_relevant_missing_attr_token = 0
    irrelevant_all_token_counts: list[int] = []
    rarest_df_values: list[int] = []

    single_token_bm25 = []
    multi_token_bm25 = []
    failure_examples: list[dict] = []

    n_queries = len(queries)
    for query_number, query in enumerate(queries, start=1):
        query_id = query["_id"]
        attribute = extract_attribute(query["text"])
        rel_ids = relevant[query_id]
        rel_idx = [id_to_index[doc_id] for doc_id in rel_ids]
        rel_set = set(rel_idx)

        exact_idx = [doc_id for doc_id, text in enumerate(texts) if attribute in text]
        exact_set = set(exact_idx)
        irrelevant_with_attribute = len(exact_set - rel_set)
        irrelevant_counts.append(irrelevant_with_attribute)
        if irrelevant_with_attribute > 0:
            any_irrelevant += 1
        if rel_set <= exact_set:
            both_relevant += 1
        elif all(attribute.casefold() in texts[doc_id].casefold() for doc_id in rel_idx):
            casefold_only_both += 1

        if exact_set == rel_set:
            n_exact_equals_relevant += 1

        attr_tokens = tokenize(attribute)
        query_tokens = tokenize(query["text"])
        token_docs = docs_with_all_tokens(attr_tokens, index["postings_idx"])
        token_set = set(token_docs)
        if token_set == rel_set:
            n_token_complete_equals_relevant += 1
        irrelevant_all_token_counts.append(len(token_set - rel_set))
        attr_types = list(dict.fromkeys(attr_tokens))
        missing_token = any(
            not contains_token(index["postings_idx"], token, doc_id)
            for doc_id in rel_idx
            for token in attr_types
        )
        if missing_token:
            n_relevant_missing_attr_token += 1

        present_tokens = [token for token in attr_types if token in index["postings_idx"]]
        present_df = [int(index["postings_idx"][token].shape[0]) for token in present_tokens]
        rarest_df = min(present_df) if present_df else 0
        rarest_df_values.append(rarest_df)
        if rarest_df == 2 and not missing_token:
            n_rarest_df_eq_2 += 1
        rarest_token = (
            min(present_tokens, key=lambda token: int(index["postings_idx"][token].shape[0]))
            if present_tokens
            else None
        )

        bm25 = bm25_scores(
            query_tokens,
            index["postings_idx"],
            index["postings_tf"],
            index["idf"],
            index["doc_len"],
            index["avgdl"],
        )
        substr = np.zeros(len(doc_ids), dtype=np.float64)
        if exact_idx:
            substr[np.asarray(exact_idx, dtype=np.int32)] = 1.0

        bm25_order = rank_order(bm25, id_rank)
        substr_order = rank_order(substr, id_rank)
        bm25_rank = np.empty(len(doc_ids), dtype=np.int32)
        substr_rank = np.empty(len(doc_ids), dtype=np.int32)
        bm25_rank[bm25_order] = np.arange(len(doc_ids), dtype=np.int32)
        substr_rank[substr_order] = np.arange(len(doc_ids), dtype=np.int32)

        bm25_ranks = sorted(int(bm25_rank[doc_id]) for doc_id in rel_idx)
        substr_ranks = sorted(int(substr_rank[doc_id]) for doc_id in rel_idx)

        if rarest_token is not None:
            rare_posting = index["postings_idx"][rarest_token]
            candidate_set = set(int(doc_id) for doc_id in rare_posting)
            worse_relevant_rank = max(bm25_ranks)
            better = bm25_order[:worse_relevant_rank]
            if any(int(doc_id) not in candidate_set for doc_id in better):
                n_noncandidate_outranks += 1

        for k in ks:
            bm25_hits = sum(rank < k for rank in bm25_ranks)
            substr_hits = sum(rank < k for rank in substr_ranks)
            bm25_sum[k] += bm25_hits / 2.0
            substr_sum[k] += substr_hits / 2.0
            bm25_hit_hist[k][bm25_hits] += 1
            substr_hit_hist[k][substr_hits] += 1

        recall_at_2 = sum(rank < 2 for rank in bm25_ranks) / 2.0
        (single_token_bm25 if len(attr_types) == 1 else multi_token_bm25).append(recall_at_2)

        if recall_at_2 < 1.0 and len(failure_examples) < 12:
            top_ids = []
            for doc_id in bm25_order[:5]:
                doc_index = int(doc_id)
                top_ids.append(
                    {
                        "doc_id": doc_ids[doc_index],
                        "score": round(float(bm25[doc_index]), 6),
                        "exact_attribute_substring": doc_index in exact_set,
                        "relevant": doc_index in rel_set,
                        "doc_len_tokens": int(index["doc_len"][doc_index]),
                    }
                )
            failure_examples.append(
                {
                    "query_id": query_id,
                    "attribute": attribute,
                    "attribute_tokens": list(dict.fromkeys(attr_tokens)),
                    "rarest_attribute_token": rarest_token,
                    "rarest_attribute_token_df": rarest_df,
                    "relevant": [
                        {
                            "doc_id": doc_ids[doc_index],
                            "rank": int(bm25_rank[doc_index]),
                            "score": round(float(bm25[doc_index]), 6),
                            "exact_attribute_substring": doc_index in exact_set,
                            "doc_len_tokens": int(index["doc_len"][doc_index]),
                        }
                        for doc_index in rel_idx
                    ],
                    "bm25_top5": top_ids,
                    "n_docs_with_exact_substring": len(exact_set),
                    "n_docs_with_all_attribute_tokens": len(token_set),
                }
            )

        if query_number % 250 == 0:
            print(f"[{name}] scored {query_number}/{n_queries} queries", flush=True)

    def percent(total: float) -> float:
        return 100.0 * total / n_queries

    def mean_or_none(values: list[float]) -> float | None:
        if not values:
            return None
        return 100.0 * float(np.mean(values))

    bm25_recall = {f"recall@{k}": percent(bm25_sum[k]) for k in ks}
    substr_recall = {f"recall@{k}": percent(substr_sum[k]) for k in ks}
    paper = PAPER_RECALL[name]
    paper_delta = {
        f"recall@{k}": bm25_recall[f"recall@{k}"] - paper[k] for k in ks
    }

    return {
        "dataset": name,
        "n_docs": len(corpus),
        "n_queries": n_queries,
        "n_qrel_rows": len(qrel_rows),
        "empty_titles": sum(title == "" for title in titles),
        "index": {
            "vocab_size": index["vocab_size"],
            "avgdl_tokens": index["avgdl"],
            "mean_doc_chars": float(np.mean([len(text) for text in texts])),
        },
        "attribute_occurrence": {
            "fraction_queries_attribute_in_both_relevant_docs": both_relevant / n_queries,
            "fraction_queries_attribute_in_any_irrelevant_doc": any_irrelevant / n_queries,
            "n_queries_attribute_in_both_relevant_docs": both_relevant,
            "n_queries_attribute_in_any_irrelevant_doc": any_irrelevant,
            "n_queries_both_relevant_only_after_casefold": casefold_only_both,
            "n_queries_exact_substring_set_equals_relevant_set": n_exact_equals_relevant,
            "irrelevant_docs_containing_attribute": distribution(irrelevant_counts),
        },
        "token_overlap": {
            "n_queries_all_attribute_tokens_set_equals_relevant_set": n_token_complete_equals_relevant,
            "n_queries_relevant_doc_missing_an_attribute_token": n_relevant_missing_attr_token,
            "n_queries_rarest_attribute_token_df_eq_2": n_rarest_df_eq_2,
            "n_queries_doc_without_rarest_token_outranks_relevant": n_noncandidate_outranks,
            "irrelevant_docs_containing_all_attribute_tokens": distribution(
                irrelevant_all_token_counts
            ),
            "rarest_attribute_token_df": distribution(rarest_df_values),
            "bm25_recall@2_single_attribute_token_percent": mean_or_none(single_token_bm25),
            "bm25_recall@2_multi_attribute_token_percent": mean_or_none(multi_token_bm25),
            "n_single_attribute_token_queries": len(single_token_bm25),
            "n_multi_attribute_token_queries": len(multi_token_bm25),
        },
        "bm25_recall_percent": bm25_recall,
        "substring_recall_percent": substr_recall,
        "paper_bm25_recall_percent": {f"recall@{k}": paper[k] for k in ks},
        "bm25_minus_paper_percentage_points": paper_delta,
        "bm25_relevant_hit_counts": {
            f"@{k}": {str(hits): count for hits, count in sorted(hist.items())}
            for k, hist in bm25_hit_hist.items()
        },
        "substring_relevant_hit_counts": {
            f"@{k}": {str(hits): count for hits, count in sorted(hist.items())}
            for k, hist in substr_hit_hist.items()
        },
        "bm25_recall_at_2_failure_examples": failure_examples,
        "_relevant_pairs": {
            query["_id"]: tuple(sorted(relevant[query["_id"]])) for query in queries
        },
        "_query_text": {query["_id"]: query["text"] for query in queries},
    }


def qrel_graph(pair_map: dict[str, tuple[str, str]]) -> dict:
    edges: set[tuple[str, str]] = set()
    queries_per_person: Counter[str] = Counter()
    queries_per_edge: Counter[tuple[str, str]] = Counter()
    for _query_id, pair in pair_map.items():
        if len(pair) != 2:
            raise AssertionError(pair)
        edge = tuple(sorted(pair))
        edges.add(edge)  # type: ignore[arg-type]
        queries_per_edge[edge] += 1  # type: ignore[index]
        queries_per_person[pair[0]] += 1
        queries_per_person[pair[1]] += 1

    nodes = sorted(queries_per_person)
    n_nodes = len(nodes)
    n_edges = len(edges)
    possible = n_nodes * (n_nodes - 1) / 2.0
    per_person = np.asarray([queries_per_person[node] for node in nodes], dtype=np.float64)
    degrees = Counter()
    for left, right in edges:
        degrees[left] += 1
        degrees[right] += 1
    degree_values = np.asarray([degrees[node] for node in nodes], dtype=np.float64)
    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_query_pair_instances": len(pair_map),
        "n_edges_with_multiple_queries": sum(count > 1 for count in queries_per_edge.values()),
        "density": n_edges / possible,
        "possible_edges": int(possible),
        "mean_queries_per_person": float(per_person.mean()),
        "min_queries_per_person": int(per_person.min()),
        "max_queries_per_person": int(per_person.max()),
        "mean_unique_partners_per_person": float(degree_values.mean()),
        "definition": (
            "Undirected simple graph on people who appear in at least one qrel. "
            "An edge exists when at least one query lists that pair as its two relevant documents. "
            "density = |E| / C(|V|, 2). "
            "mean_queries_per_person is the average, over people, of how many queries include that person."
        ),
    }


def _self_test() -> None:
    # Hand-checked score: query token "bb" against doc "aa bb" (len 2) and "aa" (len 1).
    texts = ["aa bb", "aa"]
    index = build_index(texts)
    scores = bm25_scores(
        ["bb"],
        index["postings_idx"],
        index["postings_tf"],
        index["idf"],
        index["doc_len"],
        index["avgdl"],
    )
    expected_idf = math.log(1.0 + (2 - 1 + 0.5) / (1 + 0.5))
    # avgdl = 1.5, dl = 2, tf = 1
    norm = 1.0 - B + B * (2.0 / 1.5)
    expected = expected_idf * (1.0 * (K1 + 1.0)) / (1.0 + K1 * norm)
    if not math.isclose(scores[0], expected, rel_tol=1e-12):
        raise AssertionError((scores[0], expected))
    if scores[1] != 0.0:
        raise AssertionError(scores[1])

    # Equal content scores: lexicographically smaller _id wins.
    ids = np.asarray(["Bob", "Ann"])
    id_rank = np.empty(2, dtype=np.int32)
    id_rank[np.argsort(ids, kind="mergesort")] = np.arange(2)
    order = rank_order(np.asarray([1.0, 1.0]), id_rank)
    if ids[order].tolist() != ["Ann", "Bob"]:
        raise AssertionError(ids[order])

    # Shorter document with the same rare term ranks higher.
    length_index = build_index(
        [
            "short likes zebra ants and bees",
            "long likes zebra " + " ".join(f"item{i}" for i in range(40)),
        ]
    )
    length_scores = bm25_scores(
        tokenize("Who likes zebra?"),
        length_index["postings_idx"],
        length_index["postings_tf"],
        length_index["idf"],
        length_index["doc_len"],
        length_index["avgdl"],
    )
    if not length_scores[0] > length_scores[1]:
        raise AssertionError(length_scores)

    if tokenize("Who likes Rubik's Cubes?") != ["who", "likes", "rubik", "s", "cubes"]:
        raise AssertionError(tokenize("Who likes Rubik's Cubes?"))
    if extract_attribute("Who likes the Pittsburgh Pirates?") != "the Pittsburgh Pirates":
        raise AssertionError(extract_attribute("Who likes the Pittsburgh Pirates?"))


def main() -> None:
    _self_test()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    small = evaluate_dataset("limit-small")
    full = evaluate_dataset("limit")

    def people_of(pair_map: dict[str, tuple[str, str]]) -> set[str]:
        found: set[str] = set()
        for left, right in pair_map.values():
            found.add(left)
            found.add(right)
        return found

    def attributes_of(query_text: dict[str, str]) -> set[str]:
        return {extract_attribute(text) for text in query_text.values()}

    small_people = people_of(small["_relevant_pairs"])
    full_people = people_of(full["_relevant_pairs"])
    small_attrs = attributes_of(small["_query_text"])
    full_attrs = attributes_of(full["_query_text"])
    alignment = {
        "query_ids_identical": set(small["_query_text"]) == set(full["_query_text"]),
        "query_texts_identical": small["_query_text"] == full["_query_text"],
        "relevant_pairs_identical": small["_relevant_pairs"] == full["_relevant_pairs"],
        "n_shared_query_texts": len(set(small["_query_text"].values()) & set(full["_query_text"].values())),
        "n_shared_attributes": len(small_attrs & full_attrs),
        "n_shared_relevant_people": len(small_people & full_people),
        "n_limit_small_people_in_limit_corpus": None,
        "note": (
            "These local files do not match the usual LIMIT construction in which limit-small "
            "is the 46 relevant documents of the same 1000 queries. Query strings, relevant "
            "people, and relevant pairs differ, so each split is evaluated with its own queries, "
            "corpus, and qrels."
        ),
    }
    full_ids = set()
    with (DATA_ROOT / "limit" / "corpus.jsonl").open() as handle:
        for line in handle:
            full_ids.add(json.loads(line)["_id"])
    alignment["n_limit_small_people_in_limit_corpus"] = len(small_people & full_ids)

    graph = {
        "limit-small": qrel_graph(small["_relevant_pairs"]),
        "limit": qrel_graph(full["_relevant_pairs"]),
    }
    for result in (small, full):
        result.pop("_relevant_pairs")
        result.pop("_query_text")

    command = f"{sys.executable} {Path(__file__).resolve()}"
    payload = {
        "command": command,
        "python": sys.version,
        "numpy": np.__version__,
        "definitions": {
            "attribute": (
                "The raw query span after the prefix 'Who likes ' and before the final '?'. "
                "Case is preserved."
            ),
            "attribute_occurs": (
                "Case-sensitive Python substring test of that attribute against the document text. "
                "Titles are empty and are not searched."
            ),
            "irrelevant_doc": "A corpus document whose _id is not one of the query's two qrel corpus-ids.",
            "tokenizer": (
                "Lowercase, then maximal [a-z0-9]+ runs. Whitespace and punctuation are separators "
                "and are not tokens."
            ),
            "bm25": (
                "score(q, d) = sum_t idf(t) * tf(t,d) * (k1 + 1) / "
                "(tf(t,d) + k1 * (1 - b + b * |d| / avgdl)), "
                "idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5)), "
                f"k1={K1}, b={B}. Each query term type contributes once. "
                "Document length and df are computed inside the corpus being evaluated. "
                "Terms absent from the corpus contribute nothing."
            ),
            "substring_baseline": (
                "score 1 when the raw attribute is a substring of the document text, else 0."
            ),
            "tie_break": "Sort by score descending, then by document _id ascending (byte/Unicode lexicographic).",
            "recall": (
                "Macro-average over the 1000 queries of |top-k intersect relevant| / 2, "
                "reported as a percentage (paper scale, 100 = perfect)."
            ),
            "qrel_graph": (
                "46 relevant people as nodes. Undirected edge if some query has that pair as its "
                "two relevant documents. density = |E| / C(46, 2). "
                "mean_queries_per_person averages, over people, the number of queries that list them."
            ),
        },
        "dataset_alignment": alignment,
        "qrel_graph": graph,
        "datasets": {"limit-small": small, "limit": full},
    }
    out_path = OUT_DIR / "results.json"
    with out_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
