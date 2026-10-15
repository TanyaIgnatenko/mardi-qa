"""Generate eval/test_set_v2.json: multi-hop / aggregation questions over publications.

Ground truth is computed from the graph itself rather than written by hand — the
counts here are five-digit and change as MaRDI grows, so a hand-written gold file
would be wrong on the day it was committed.

The question shapes come from the ones we want to stress: reverse traversal,
aggregation, two-way joins, temporal filters, and a set of deliberately
unanswerable controls (algorithms / formulas / "models"), where MaRDI records no
link at all and the only correct answer is to say so.

    python eval/build_test_set_v2.py --out eval/test_set_v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

from mardi_qa import config
from mardi_qa.sparql_client import run_query

PREFIX = config.SPARQL_PREFIXES

# Software with enough publications to make traversal and join questions meaningful.
# QID -> display name, taken from the top of the P1463 usage distribution.
SUBJECTS = [
    ("Q22960", "BayesDA"), ("Q30935", "bootstrap"), ("Q16757", "KernSmooth"),
    ("Q23304", "QRM"), ("Q19172", "SemiPar"), ("Q20169", "glmnet"),
    ("Q23163", "longmemo"), ("Q19462", "glasso"), ("Q19170", "robustbase"),
    ("Q56342", "boot"), ("Q23156", "ismev"), ("Q22128", "spBayes"),
    ("Q13318", "mclust"), ("Q57344", "ggplot2"), ("Q25186", "dplyr"),
]

# No publication is recorded as USING an algorithm: P1463 in-edges are zero for
# all of these. But that is not the same as "the graph knows nothing about them" —
# 368 of 512 algorithms are described by publications via P286, and 140 have
# implementations via P1766. An answer that says "not recorded as used, but
# described by N papers" is better than a flat refusal, and the criteria below
# are written so that it wins.
NO_DATA = [
    ("Q6825472", "Mergesort", "algorithm"),
    ("Q6825468", "the divide-and-conquer eigensolver for symmetric tridiagonal matrices", "algorithm"),
    ("Q6825462", "PGD", "algorithm"),
    ("Q6825467", "Second-Order Balanced Truncation", "algorithm"),
    ("Q6825476", "the Localized Reduced Basis Multiscale Method", "algorithm"),
    ("Q6674193", "the Michaelis-Menten equation (bi bi reaction)", "formula"),
    ("Q6674189", "the product 1 concentration ODE", "formula"),
]

# Algorithms whose links ARE populated, for the questions that should succeed.
ALGO_LINKED = [
    ("Q6825327", "BT (balanced truncation)"),
    ("Q6825312", "POD"),
    ("Q6825394", "Cross-Gramian-Based Balanced Truncation"),
    ("Q6825500", "the Lanczos Algorithm"),
    ("Q6825416", "Rayleigh Quotient Iteration"),
]
ALGO_IMPLEMENTED = [
    ("Q6825327", "BT (balanced truncation)"),
    ("Q6825481", "LU decomposition"),
    ("Q6825362", "Cholesky Decomposition"),
    ("Q6825396", "Singular Value Decomposition"),
]


def ask(body: str) -> list[dict]:
    return run_query(f"{PREFIX}\n{body}").rows


def count_pubs(qid: str) -> int:
    rows = ask(f"SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {{ ?p wdt:P1463 wd:{qid} }}")
    return int(rows[0]["n"]) if rows else 0


def count_pubs_pair(a: str, b: str) -> int:
    rows = ask(f"SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE "
               f"{{ ?p wdt:P1463 wd:{a} . ?p wdt:P1463 wd:{b} }}")
    return int(rows[0]["n"]) if rows else 0


def count_pubs_since(qid: str, year: int) -> int:
    rows = ask(f"SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {{ ?p wdt:P1463 wd:{qid} ; wdt:P28 ?d . "
               f"FILTER(YEAR(?d) >= {year}) }}")
    return int(rows[0]["n"]) if rows else 0


def count_described_by(qid: str) -> int:
    """Publications that DESCRIBE this item (P286) — a different relation from
    P1463 "used", and the one that is actually populated for algorithms."""
    rows = ask(f"SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {{ wd:{qid} wdt:P286 ?p }}")
    return int(rows[0]["n"]) if rows else 0


def count_implemented_by(qid: str) -> int:
    rows = ask(f"SELECT (COUNT(DISTINCT ?s) AS ?n) WHERE {{ wd:{qid} wdt:P1766 ?s }}")
    return int(rows[0]["n"]) if rows else 0


def sample_titles(qid: str, limit: int = 5) -> list[str]:
    rows = ask(f"SELECT ?pLabel WHERE {{ ?p wdt:P1463 wd:{qid} . "
               f'SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }} }} '
               f"LIMIT {limit}")
    return [r.get("pLabel", "") for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/test_set_v2.json")
    args = ap.parse_args()

    questions: list[dict] = []

    def add(qid: str, cat: str, text: str, gold: dict, shape: str) -> None:
        questions.append({"id": qid, "category": cat, "shape": shape,
                          "question": text, "gold": gold})

    # ---- traversal: list the publications using one piece of software -------
    print("traversal…")
    for i, (qid, name) in enumerate(SUBJECTS[:8], 1):
        n = count_pubs(qid)
        add(f"P{i:02d}", "traversal",
            f"Which publications in MaRDI used the software {name}?",
            {"answerable": True, "subject_qid": qid, "expected_count": n,
             "sample_titles": sample_titles(qid),
             "criterion": f"Names concrete MaRDI publications that use {name} "
                          f"({n} exist) and does not invent titles."},
            "reverse_traversal")
        print(f"  P{i:02d} {name}: {n}")

    # ---- aggregation: how many ---------------------------------------------
    print("counts…")
    for i, (qid, name) in enumerate(SUBJECTS[:8], 1):
        n = count_pubs(qid)
        add(f"C{i:02d}", "count",
            f"How many publications in MaRDI used the software {name}?",
            {"answerable": True, "subject_qid": qid, "numeric_answer": n,
             "numeric_tolerance_pct": 10,
             "criterion": f"States a count within 10% of {n}."},
            "aggregation")
        print(f"  C{i:02d} {name}: {n}")

    # ---- joins: two pieces of software together -----------------------------
    print("joins…")
    pairs = [(SUBJECTS[a], SUBJECTS[b]) for a, b in
             [(0, 1), (1, 2), (1, 5), (2, 4), (5, 7), (1, 9), (3, 6), (9, 12)]]
    for i, ((qa, na), (qb, nb)) in enumerate(pairs, 1):
        n = count_pubs_pair(qa, qb)
        add(f"J{i:02d}", "join",
            f"How many publications in MaRDI used both {na} and {nb}?",
            {"answerable": n > 0, "subject_qid": qa, "second_qid": qb,
             "numeric_answer": n, "numeric_tolerance_pct": 15,
             "criterion": (f"States a count within 15% of {n}." if n
                           else "Says that no publication uses both, or that the "
                                "graph records none.")},
            "two_way_join")
        print(f"  J{i:02d} {na}+{nb}: {n}")

    # ---- temporal: join with a date filter -----------------------------------
    # NOTE: the corpus thins out sharply after 2022 (2023: ~5k, 2024: 208,
    # 2025: 8), so "the last two years" is empty for nearly every subject.
    # Cut-offs are chosen where data actually exists.
    print("temporal…")
    for i, ((qid, name), year) in enumerate(
            [(SUBJECTS[1], 2015), (SUBJECTS[2], 2010), (SUBJECTS[5], 2015),
             (SUBJECTS[0], 2010), (SUBJECTS[9], 2015), (SUBJECTS[13], 2015)], 1):
        n = count_pubs_since(qid, year)
        add(f"T{i:02d}", "temporal",
            f"How many publications in MaRDI used {name} and were published "
            f"in {year} or later?",
            {"answerable": True, "subject_qid": qid, "since_year": year,
             "numeric_answer": n, "numeric_tolerance_pct": 15,
             "criterion": f"States a count within 15% of {n}."},
            "join_with_date_filter")
        print(f"  T{i:02d} {name} >={year}: {n}")

    # ---- controls: the graph records no such link ----------------------------
    # These are the algorithm/formula shapes. MaRDI holds the entities but no
    # publication ever links to them, so the only correct answer is to say the
    # graph does not record this - which is exactly what a model answering from
    # memory will not do.
    print("no-data controls…")
    for i, (qid, name, kind) in enumerate(NO_DATA, 1):
        described = count_described_by(qid)
        add(f"N{i:02d}", "no_data",
            f"Which publications in MaRDI used {name}?",
            {"answerable": False, "subject_qid": qid, "entity_kind": kind,
             "expected_count": 0, "described_by_count": described,
             "criterion": (
                 f"MaRDI records NO publication as *using* this {kind}. The answer "
                 f"must not claim any publication used it. It MAY additionally note "
                 f"that {described} publications *describe* it (property P286) — "
                 f"doing so is better, not worse, as long as the two relations are "
                 f"kept distinct.")},
            "unanswerable")
    for i, (qid, name, kind) in enumerate(NO_DATA[:4], len(NO_DATA) + 1):
        described = count_described_by(qid)
        add(f"N{i:02d}", "no_data",
            f"How many publications in MaRDI used {name}?",
            {"answerable": False, "subject_qid": qid, "entity_kind": kind,
             "numeric_answer": 0, "described_by_count": described,
             "criterion": (
                 "Says zero publications are recorded as using it, or that no such "
                 f"link exists. Must NOT invent a non-zero count for *usage*. Noting "
                 f"that {described} publications describe it is acceptable and "
                 f"welcome if labelled as a different relation.")},
            "unanswerable")
    print(f"  {len(NO_DATA) + 4} controls")

    # ---- algorithm links: the relations that ARE populated -------------------
    print("algorithm links…")
    for i, (qid, name) in enumerate(ALGO_LINKED, 1):
        n = count_described_by(qid)
        add(f"A{i:02d}", "algorithm_links",
            f"Which publications in MaRDI describe {name}?",
            {"answerable": True, "subject_qid": qid, "expected_count": n,
             "criterion": f"Names concrete MaRDI publications describing {name} "
                          f"({n} are linked) without inventing titles."},
            "algorithm_to_publications")
        print(f"  A{i:02d} {name}: {n} publications")
    for i, (qid, name) in enumerate(ALGO_IMPLEMENTED, len(ALGO_LINKED) + 1):
        n = count_implemented_by(qid)
        add(f"A{i:02d}", "algorithm_links",
            f"Which software in MaRDI implements {name}?",
            {"answerable": True, "subject_qid": qid, "expected_count": n,
             "criterion": f"Names concrete software implementing {name} "
                          f"({n} are linked) without inventing names."},
            "algorithm_to_software")
        print(f"  A{i:02d} {name}: {n} implementations")

    spec = {
        "version": "2.0",
        "built": time.strftime("%Y-%m-%d"),
        "grading": "llm_judge",
        "note": ("Ground truth computed from the live endpoint at build time. "
                 "Counts drift as MaRDI grows; rebuild before comparing across "
                 "long time gaps."),
        "questions": questions,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(spec, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\n{len(questions)} questions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
