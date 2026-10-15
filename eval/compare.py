"""Compare two or more regression reports side by side.

Intended use: put the full pipeline next to the naive control, so the headline
number has something to be measured against.

    python eval/run_regression.py --mode naive --out eval/reports/naive.json
    python eval/run_regression.py --mode full  --out eval/reports/full.json
    python eval/compare.py eval/reports/naive.json eval/reports/full.json

The last report given is treated as the system under test; the others are
references. Questions the references pass and it does not are called out
separately — those are the cases where the pipeline costs more and delivers
less, which is the failure mode a single pass rate hides.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: str) -> dict:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    report["_name"] = report.get("mode", "?")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="+", help="report JSONs; last = system under test")
    ap.add_argument("--full", action="store_true", help="list every question, not just diffs")
    args = ap.parse_args()

    reports = [load(p) for p in args.reports]

    versions = {r["test_set_version"] for r in reports}
    if len(versions) > 1:
        print(f"WARNING: reports span test set versions {sorted(versions)} - "
              f"pass rates are not comparable.\n", file=sys.stderr)
    for r, path in zip(reports, args.reports):
        if r.get("partial"):
            print(f"WARNING: {path} is a partial run ({r['total']} questions).\n",
                  file=sys.stderr)

    # ------------------------------------------------------------- headline ---
    width = max(len(f"{r['_name']} ({r['model']})") for r in reports)
    print("=" * (width + 22))
    for r in reports:
        pct = 100.0 * r["passed"] / r["total"] if r["total"] else 0.0
        label = f"{r['_name']} ({r['model']})"
        print(f"{label:<{width}}  {r['passed']:>3}/{r['total']:<3}  {pct:5.1f}%")
    print("=" * (width + 22))

    # ------------------------------------------------------------ per test ---
    passes = [{x["id"]: x["passed"] for x in r["results"]} for r in reports]
    cats = {x["id"]: x["category"] for r in reports for x in r["results"]}
    ids = sorted({i for p in passes for i in p}, key=lambda i: (cats[i], i))

    # ---------------------------------------------------------- by category ---
    print("\nBy category:")
    names = [r["_name"] for r in reports]
    print(f"  {'category':<14} " + "  ".join(f"{n:>10}" for n in names))
    for cat in sorted(set(cats.values())):
        cells = []
        for p in passes:
            got = [p[i] for i in ids if cats[i] == cat and i in p]
            cells.append(f"{sum(got)}/{len(got)}" if got else "-")
        print(f"  {cat:<14} " + "  ".join(f"{c:>10}" for c in cells))

    # --------------------------------------------------------------- deltas ---
    under_test, refs = passes[-1], passes[:-1]
    lost = [i for i in ids
            if i in under_test and not under_test[i]
            and any(p.get(i) for p in refs)]
    won = [i for i in ids
           if under_test.get(i)
           and any(i in p and not p[i] for p in refs)]

    print(f"\nOnly {reports[-1]['_name']} passes ({len(won)}): "
          f"{', '.join(won) if won else 'none'}")
    print(f"Only a reference passes ({len(lost)}): "
          f"{', '.join(lost) if lost else 'none'}")

    if args.full:
        print("\nPer question:")
        print(f"  {'id':<6} {'category':<14} " + "  ".join(f"{n:>10}" for n in names))
        for i in ids:
            marks = ["-" if i not in p else "PASS" if p[i] else "FAIL" for p in passes]
            print(f"  {i:<6} {cats[i]:<14} " + "  ".join(f"{m:>10}" for m in marks))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
