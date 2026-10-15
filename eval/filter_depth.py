"""Does answer quality degrade as filter conditions are added?

Matched design: the same subjects asked at four levels of conjunction, so the
only thing that varies within a row is how many conditions must hold at once.

    F1   one software filter
    F2   two software filters
    F1T  one software filter + a year bound
    F2T  two software filters + a year bound

Questions deliberately do NOT mention MaRDI and never show a QID: we want to see
what each system does with a plain bibliometric question, including whether the
baseline invents identifiers unprompted.

Scoring here is objective, not an LLM judge - for this question the measures are
mechanical: did it commit to a number, was that number right, did it emit QIDs
that do not exist.

    python eval/filter_depth.py --out eval/reports/filter_depth.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

from mardi_qa import config
from mardi_qa.entity_linking import fetch_property_catalog
from mardi_qa.gateway import route_and_answer
from mardi_qa.llm import LLM
from mardi_qa.sparql_client import run_query
from run_regression import answer_naive

# (qid_a, name_a, qid_b, name_b, year)
ROWS = [
    ("Q20169", "glmnet",     "Q19462", "glasso",     2015),
    ("Q16757", "KernSmooth", "Q19172", "SemiPar",    2010),
    ("Q22960", "BayesDA",    "Q30935", "bootstrap",  2010),
    ("Q56342", "boot",       "Q13318", "mclust",     2015),
    ("Q23304", "QRM",        "Q20074", "sn",         2010),
]


def count(where: str) -> int:
    rows = run_query(f"{config.SPARQL_PREFIXES}\n"
                     f"SELECT (COUNT(DISTINCT ?pub) AS ?n) WHERE {{ {where} }}").rows
    return int(rows[0]["n"]) if rows else 0


def gold_for(a: str, b: str, year: int) -> dict[str, int]:
    ya = f"?pub wdt:P28 ?d . FILTER(YEAR(?d) >= {year})"
    return {
        "F1":  count(f"?pub wdt:P1463 wd:{a} ."),
        "F2":  count(f"?pub wdt:P1463 wd:{a} . ?pub wdt:P1463 wd:{b} ."),
        "F1T": count(f"?pub wdt:P1463 wd:{a} . {ya}"),
        "F2T": count(f"?pub wdt:P1463 wd:{a} . ?pub wdt:P1463 wd:{b} . {ya}"),
    }


def phrase(level: str, na: str, nb: str, year: int) -> str:
    return {
        "F1":  f"How many publications used the R package {na}?",
        "F2":  f"How many publications used both the R packages {na} and {nb}?",
        "F1T": f"How many publications used the R package {na} and were "
               f"published in {year} or later?",
        "F2T": f"How many publications used both the R packages {na} and {nb} "
               f"and were published in {year} or later?",
    }[level]


REFUSAL = re.compile(
    r"\b(cannot|can't|don't have|do not have|no access|unable|not able|"
    r"would need to|i'm not able|no way to know|returned no)\b", re.I)


def analyse(answer: str, gold: int, tol: float = 0.15) -> dict:
    """Objective read of one answer: committed number, correctness, fake QIDs."""
    text = answer or ""
    # Numbers that plausibly answer "how many": ignore years and small list markers.
    nums = [int(n.replace(",", "")) for n in re.findall(r"\b\d[\d,]{0,9}\b", text)]
    nums = [n for n in nums if not (1900 <= n <= 2100)]
    hit = any(abs(n - gold) <= max(1, gold * tol) for n in nums) if gold else (0 in nums)
    return {
        "gold": gold,
        "committed": bool(nums),
        "closest": min(nums, key=lambda n: abs(n - gold)) if nums else None,
        "correct": hit,
        "refused": bool(REFUSAL.search(text)),
        "qids_emitted": sorted(set(re.findall(r"\bQ\d{2,7}\b", text))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/reports/filter_depth.json")
    ap.add_argument("--levels", default="F1,F2,F1T,F2T")
    args = ap.parse_args()
    levels = [s.strip() for s in args.levels.split(",")]

    llm = LLM()
    print(f"Model: {llm.provider}/{llm.model}\nComputing ground truth…")
    golds = {a: gold_for(a, b, y) for a, _, b, _, y in ROWS}
    catalog = fetch_property_catalog()

    results = []
    for a, na, b, nb, year in ROWS:
        for lvl in levels:
            qtext = phrase(lvl, na, nb, year)
            g = golds[a][lvl]
            print(f"\n[{na}/{lvl}] {qtext}\n   gold={g}", flush=True)
            row = {"subject": na, "second": nb, "level": lvl, "year": year,
                   "question": qtext, "gold": g}
            for branch, fn in (("naive", lambda: answer_naive(qtext, llm)),
                               ("full", lambda: route_and_answer(qtext, llm, catalog))):
                t = time.time()
                try:
                    tr = fn()
                    ans, route, err = tr.answer, tr.route, None
                except Exception as exc:
                    ans, route, err = "", None, f"{type(exc).__name__}: {exc}"
                row[branch] = {**analyse(ans, g), "route": route, "error": err,
                               "latency_s": round(time.time() - t, 2),
                               "answer": ans}
                m = row[branch]
                print(f"   {branch:<5} n={str(m['closest']):>7} "
                      f"{'OK ' if m['correct'] else 'BAD'} "
                      f"refused={int(m['refused'])} qids={m['qids_emitted'] or '-'} "
                      f"[{route}]")
            results.append(row)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(
                {"model": f"{llm.provider}/{llm.model}",
                 "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
                 "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")

    _summary(results, levels)
    print(f"\nReport: {args.out}")
    return 0


def _summary(results: list[dict], levels: list[str]) -> None:
    print(f"\n{'=' * 70}\nCorrect answers by number of conditions\n")
    print(f"  {'level':<5} {'conditions':<12} {'naive':>10} {'full':>10}")
    names = {"F1": "1 (software)", "F2": "2 (sw+sw)",
             "F1T": "2 (sw+year)", "F2T": "3 (sw+sw+year)"}
    for lvl in levels:
        rows = [r for r in results if r["level"] == lvl]
        if not rows:
            continue
        nv = sum(r["naive"]["correct"] for r in rows)
        fl = sum(r["full"]["correct"] for r in rows)
        print(f"  {lvl:<5} {names.get(lvl, ''):<12} {nv:>5}/{len(rows):<4} {fl:>5}/{len(rows):<4}")

    print("\nBehaviour of the baseline:")
    for lvl in levels:
        rows = [r for r in results if r["level"] == lvl]
        if not rows:
            continue
        comm = sum(r["naive"]["committed"] for r in rows)
        ref = sum(r["naive"]["refused"] for r in rows)
        fake = sum(len(r["naive"]["qids_emitted"]) > 0 for r in rows)
        print(f"  {lvl:<5} committed to a number {comm}/{len(rows)}   "
              f"refused {ref}/{len(rows)}   emitted QIDs {fake}/{len(rows)}")


if __name__ == "__main__":
    raise SystemExit(main())
