"""Run the regression test set against ONE model and grade it deterministically.

Purpose: detect quality drift when the system changes (prompts, entity linking,
SPARQL/schema selection). No LLM judge is involved, so a run is cheap, fast and
reproducible - suitable for running on every meaningful change.

Usage:
    python eval/run_regression.py
    python eval/run_regression.py --provider ollama --model gemma3:27b
    python eval/run_regression.py --only D01,D02      # just the regression targets
    python eval/run_regression.py --baseline eval/baseline.json   # compare vs baseline

Environment: same as the app (ANTHROPIC_API_KEY, or OLLAMA_* for local models).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# Force UTF-8 on stdout/stderr. When output is redirected to a file on Windows,
# the default console codepage cannot encode non-ASCII characters; a failure
# while printing an exception traceback kills the interpreter ("lost sys.stderr",
# exit 120) and the whole run is lost.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load ANTHROPIC_API_KEY from a gitignored .env in the project root, if present,
# so the key never has to be typed on a command line (where it lands in shell
# history). A real environment variable still wins over the file.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

from mardi_qa import prompts
from mardi_qa.entity_linking import fetch_property_catalog
from mardi_qa.gateway import route_and_answer
from mardi_qa.llm import LLM
from mardi_qa.pipeline import PipelineTrace

TEST_SET = Path(__file__).parent / "test_set.json"


# ---------------------------------------------------------------- baseline ---

def answer_naive(question: str, llm: LLM) -> PipelineTrace:
    """Control condition: ask the model and print what it says.

    No routing, no entity linking, no retrieval — and deliberately none of the
    honesty guardrails from DIRECT_ANSWER_SYSTEM, which would otherwise hand the
    baseline the very behaviour the honesty tests are there to measure. This is
    the number the full pipeline has to beat to justify its cost.
    """
    trace = PipelineTrace()
    trace.route = "naive"
    trace.route_reason = "baseline mode: no retrieval"
    trace.answer = llm.complete(prompts.NAIVE_BASELINE_SYSTEM, question, max_tokens=1000)
    return trace


# ---------------------------------------------------------------- grading ---

# Gold phrases are written out in full ("does not", "cannot"), but models freely
# write "doesn't" / "can't". Without this the honesty checks measure a model's
# contraction habit rather than whether it actually disclaimed.
_CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "won't": "will not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "hasn't": "has not", "haven't": "have not", "hadn't": "had not",
    "couldn't": "could not", "wouldn't": "would not", "shouldn't": "should not",
    "i'm": "i am", "it's": "it is", "there's": "there is",
}


def normalize(text: str) -> str:
    """Lowercase and expand contractions so substring checks compare meaning."""
    out = (text or "").lower().replace("’", "'")  # curly apostrophe -> ASCII
    for short, long in _CONTRACTIONS.items():
        out = out.replace(short, long)
    return out


def grade(question: dict, answer: str, used_qids: set[str]) -> dict:
    """Check one answer against its gold facts. Returns a per-check verdict."""
    gold = question.get("gold", {})
    answer_lc = normalize(answer)
    checks: list[dict] = []

    # 1. Required QIDs must appear in the answer or in the retrieved rows.
    for qid in gold.get("qids", []):
        hit = qid in used_qids or qid.lower() in answer_lc
        checks.append({"check": f"qid:{qid}", "passed": hit})

    # 2. Required substrings (case-insensitive).
    for phrase in gold.get("must_mention", []):
        checks.append({
            "check": f"mentions:{phrase}",
            "passed": normalize(phrase) in answer_lc,
        })

    # 3. Forbidden substrings - these catch fabrication and false "no data" claims.
    for phrase in gold.get("must_not_mention", []):
        checks.append({
            "check": f"avoids:{phrase}",
            "passed": normalize(phrase) not in answer_lc,
        })

    # 4. At least one of these must appear. Used for out_of_scope questions, where
    #    the answer must carry SOME disclaimer rather than a confident invention.
    if gold.get("must_mention_any"):
        options = gold["must_mention_any"]
        checks.append({
            "check": f"mentions_any:{'|'.join(options)}",
            "passed": any(normalize(o) in answer_lc for o in options),
        })

    # 5. No large figure may appear - catches invented statistics.
    if gold.get("must_not_contain_number"):
        found = re.findall(r"\b\d[\d,]{3,}\b", answer or "")
        checks.append({
            "check": "no_invented_number",
            "passed": not found,
            "found": found,
        })

    # 6. Numeric answers with a tolerance (counts drift as the KG grows).
    if "numeric_answer" in gold:
        expected = gold["numeric_answer"]
        tol = gold.get("numeric_tolerance_pct", 10) / 100.0
        nums = [int(n.replace(",", "")) for n in re.findall(r"\b\d[\d,]{2,}\b", answer or "")]
        hit = any(abs(n - expected) <= expected * tol for n in nums)
        checks.append({
            "check": f"numeric~{expected}(+-{int(tol * 100)}%)",
            "passed": hit,
            "found": nums,
        })

    # An out_of_scope question with no positive checks passes by not fabricating;
    # the must_not_mention entries carry the real signal there.
    passed = all(c["passed"] for c in checks) if checks else True
    return {"passed": passed, "checks": checks}


def collect_qids(trace) -> set[str]:
    """Every QID the pipeline actually touched: linked entities + result rows."""
    qids = {e.qid for e in trace.linked_entities}
    for result in trace.results:
        for row in result.rows:
            for value in row.values():
                qids.update(re.findall(r"\bQ\d+\b", str(value)))
    return qids


# ------------------------------------------------------------------- main ---

def _write_report(out: str, llm, spec: dict, results: list[dict], partial: bool,
                  mode: str = "full") -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({
        "model": f"{llm.provider}/{llm.model}",
        "mode": mode,
        "test_set_version": spec["version"],
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        "partial": partial,
        "passed": sum(r["passed"] for r in results),
        "total": len(results),
        "results": results,
    }, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None, help="anthropic (default) or ollama")
    ap.add_argument("--model", default=None)
    ap.add_argument("--only", default=None, help="comma-separated question ids")
    ap.add_argument("--mode", default="full", choices=["full", "naive"],
                    help="full = routed KG pipeline; naive = plain LLM control")
    ap.add_argument("--out", default=None, help="write full JSON report here")
    ap.add_argument("--baseline", default=None, help="baseline report to diff against")
    args = ap.parse_args()

    spec = json.loads(TEST_SET.read_text(encoding="utf-8"))
    questions = spec["questions"]
    if args.only:
        wanted = {q.strip() for q in args.only.split(",")}
        questions = [q for q in questions if q["id"] in wanted]

    # The naive control never touches the graph, so skip the slow catalog fetch.
    catalog: dict[str, str] = {}
    if args.mode == "full":
        print("Loading property catalog...")
        catalog = fetch_property_catalog()

    llm = LLM(provider=args.provider, model=args.model)
    print(f"Model: {llm.provider}/{llm.model}   Mode: {args.mode}   "
          f"Questions: {len(questions)}\n")

    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['id']}  {q['question'][:60]}", end=" ", flush=True)
        started = time.time()
        try:
            if args.mode == "naive":
                trace = answer_naive(q["question"], llm)
            else:
                trace = route_and_answer(q["question"], llm, catalog)
            verdict = grade(q, trace.answer, collect_qids(trace))
            error = None
        except Exception as exc:                      # pipeline blew up
            trace, error = None, f"{type(exc).__name__}: {exc}"
            verdict = {"passed": False, "checks": [{"check": "pipeline", "passed": False}]}

        elapsed = time.time() - started
        route = trace.route if trace else "-"
        mark = "PASS" if verdict["passed"] else "FAIL"
        print(f"{mark}  [{route}]  ({elapsed:.1f}s)")
        if not verdict["passed"]:
            for c in verdict["checks"]:
                if not c["passed"]:
                    print(f"       - failed: {c['check']}")

        results.append({
            "id": q["id"],
            "question": q["question"],
            "category": q["category"],
            "expectation": q["expectation"],
            "recorded_status": q["status"],
            "route": trace.route if trace else None,
            "route_reason": trace.route_reason if trace else None,
            "passed": verdict["passed"],
            "checks": verdict["checks"],
            "latency_s": round(elapsed, 2),
            "error": error,
            "answer": trace.answer if trace else None,
        })

        # Persist after every question: a crash mid-run then costs one question,
        # not the whole run.
        if args.out:
            _write_report(args.out, llm, spec, results, partial=True, mode=args.mode)

    # ------------------------------------------------------------ summary ---
    total, passed = len(results), sum(r["passed"] for r in results)
    print(f"\n{'=' * 60}\nPASSED {passed}/{total}")

    by_cat: dict[str, list[bool]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r["passed"])
    for cat, vals in sorted(by_cat.items()):
        print(f"  {cat:14} {sum(vals)}/{len(vals)}")

    failures = [r["id"] for r in results if not r["passed"]]
    if failures:
        print(f"\nFailing: {', '.join(failures)}")

    if args.out:
        _write_report(args.out, llm, spec, results, partial=False, mode=args.mode)
        print(f"\nReport: {args.out}")

    # ----------------------------------------------------------- baseline ---
    if args.baseline and Path(args.baseline).exists():
        base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        was = {r["id"]: r["passed"] for r in base["results"]}
        regressions = [r["id"] for r in results if was.get(r["id"]) and not r["passed"]]
        fixes = [r["id"] for r in results if was.get(r["id"]) is False and r["passed"]]
        print(f"\nvs baseline ({base['timestamp']}): "
              f"{base['passed']}/{base['total']} -> {passed}/{total}")
        if regressions:
            print(f"  REGRESSED: {', '.join(regressions)}")
        if fixes:
            print(f"  FIXED:     {', '.join(fixes)}")
        if not regressions and not fixes:
            print("  no change")
        if regressions:
            return 1

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
