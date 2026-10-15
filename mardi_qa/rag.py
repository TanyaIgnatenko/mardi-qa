"""Lightweight RAG over the R-package corpus of the MaRDI knowledge graph.

The corpus is every R package's label + English description, fetched once via
paged SPARQL and cached on disk as JSON. Retrieval is plain BM25 (pure Python,
no extra dependencies) — the corpus is ~20k short documents, for which BM25 is
fast, deterministic and embedding-free.

This backs the "rag" route of the gateway: topic questions like "which packages
are related to clustering", where the SPARQL route has to improvise fragile
mwapi full-text queries. Here we instead rank the whole corpus and can report
the true match count alongside the top examples.

CLI:
    python -m mardi_qa.rag --build        # (re)build the corpus cache
    python -m mardi_qa.rag --query "clustering"
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import config
from .sparql_client import run_query

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Words that carry no topical signal in a QUESTION ("which R packages are
# related to clustering?" -> "clustering"). Documents keep all their tokens;
# only query tokens are filtered.
_QUERY_STOPWORDS = frozenset("""
a an and are can cran deal dealing do does for i in is it me of on or
package packages r related that the there to what which with you
""".split())

# Corpus fetch: all items that are instance-of R package (Q57080).
_PAGE_SIZE = 5000
_CORPUS_QUERY = """\
SELECT ?item ?label ?desc WHERE {{
  ?item wdt:P31 wd:Q57080 .
  OPTIONAL {{ ?item rdfs:label ?label . FILTER(LANG(?label) = "en") }}
  OPTIONAL {{ ?item schema:description ?desc . FILTER(LANG(?desc) = "en") }}
}}
ORDER BY ?item
LIMIT {limit} OFFSET {offset}
"""


@dataclass
class RetrievedDoc:
    qid: str
    label: str
    description: str
    score: float


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# ----- Corpus build / cache -----------------------------------------------------


def corpus_path() -> Path:
    return Path(config.RAG_CORPUS_PATH)


def build_corpus(progress=lambda msg: None) -> list[dict]:
    """Fetch every R package (label + description) via paged SPARQL."""
    docs: list[dict] = []
    offset = 0
    while True:
        progress(f"Fetching corpus page at offset {offset}…")
        result = run_query(
            _CORPUS_QUERY.format(limit=_PAGE_SIZE, offset=offset),
            timeout=config.SPARQL_TIMEOUT_S * 2,
        )
        if result.is_empty:
            break
        for row in result.rows:
            qid = row.get("item", "")
            if not qid.startswith("Q"):
                continue
            docs.append({
                "qid": qid,
                "label": row.get("label", ""),
                "desc": row.get("desc", ""),
            })
        if len(result.rows) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE

    path = corpus_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"built": time.strftime("%Y-%m-%d"), "docs": docs}),
        encoding="utf-8",
    )
    progress(f"Corpus cached: {len(docs)} packages -> {path}")
    return docs


def load_corpus(progress=lambda msg: None) -> list[dict]:
    """Load the cached corpus, building it on first use."""
    path = corpus_path()
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["docs"]
    return build_corpus(progress)


# ----- BM25 ---------------------------------------------------------------------


class Bm25Index:
    """Okapi BM25 over (label + description) of each package."""

    K1 = 1.5
    B = 0.75

    def __init__(self, docs: list[dict]):
        self.docs = docs
        self.doc_tokens = [_tokenize(f"{d['label']} {d['desc']}") for d in docs]
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.avg_len = (sum(self.doc_lens) / len(self.doc_lens)) if docs else 0.0
        self.term_freqs = [Counter(t) for t in self.doc_tokens]
        df: Counter = Counter()
        for tf in self.term_freqs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {
            term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()
        }

    @staticmethod
    def _query_tokens(query: str) -> list[str]:
        tokens = [t for t in _tokenize(query) if t not in _QUERY_STOPWORDS]
        # If stopword removal ate everything (e.g. query "R packages"),
        # fall back to the raw tokens rather than matching nothing.
        return tokens or _tokenize(query)

    def search(self, query: str, k: int = 20) -> list[RetrievedDoc]:
        q_tokens = self._query_tokens(query)
        scores: list[float] = []
        for i, tf in enumerate(self.term_freqs):
            s = 0.0
            for term in q_tokens:
                if term not in tf:
                    continue
                f = tf[term]
                denom = f + self.K1 * (1 - self.B + self.B * self.doc_lens[i] / self.avg_len)
                s += self.idf.get(term, 0.0) * f * (self.K1 + 1) / denom
            scores.append(s)

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = []
        for i in ranked[:k]:
            if scores[i] <= 0:
                break
            d = self.docs[i]
            out.append(RetrievedDoc(d["qid"], d["label"], d["desc"], round(scores[i], 3)))
        return out

    def match_count(self, query: str) -> int:
        """How many documents share at least one query term (rough recall size)."""
        q_tokens = set(self._query_tokens(query))
        return sum(1 for tf in self.term_freqs if q_tokens & tf.keys())


_INDEX: Bm25Index | None = None


def get_index(progress=lambda msg: None) -> Bm25Index:
    """Process-wide lazy singleton (index build takes a few seconds for 20k docs)."""
    global _INDEX
    if _INDEX is None:
        progress("Loading RAG corpus…")
        docs = load_corpus(progress)
        progress(f"Building BM25 index over {len(docs)} packages…")
        _INDEX = Bm25Index(docs)
    return _INDEX


# ----- CLI ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true", help="rebuild the corpus cache")
    ap.add_argument("--query", help="run a test retrieval")
    args = ap.parse_args()

    if args.build:
        build_corpus(progress=print)
    if args.query:
        index = get_index(progress=print)
        print(f"\n~{index.match_count(args.query)} packages share a term with the query\n")
        for doc in index.search(args.query, k=15):
            print(f"  {doc.score:7.3f}  {doc.qid:9}  {doc.label}  — {doc.description[:70]}")
