"""Run test_set_v2 through both branches and score them with an LLM judge.

Why a judge here and not the substring grader: these questions have open-ended
answers (lists of publication titles, counts phrased in prose, "no such link"
disclaimers). A substring check would measure phrasing, which is exactly the
failure we already hit on v1's honesty category.

The judge is reference-based - it sees the gold criterion computed from the
graph, not just the two answers - which is far more reliable than asking a model
which answer "looks better".

    python eval/run_judge.py --out eval/reports/judge.json
    python eval/run_judge.py --only P01,C01,N01 --out eval/reports/smoke.json
"""

from __future__ import annotations

import argparse
import json
import random
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

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

from mardi_qa.entity_linking import fetch_property_catalog
from mardi_qa.gateway import route_and_answer
from mardi_qa.llm import LLM
from run_regression import answer_naive  # same control condition as v1

TEST_SET = Path(__file__).parent / "test_set_v2.json"

JUDGE_SYSTEM = """\
You grade answers produced by two question-answering systems built on the MaRDI
knowledge graph. You are given the question, a gold criterion derived directly
from the graph, and two candidate answers labelled A and B.

Grade each answer INDEPENDENTLY against the criterion. Do not compare them when
deciding pass/fail; compare them only for the preference field.

Rules:
- An answer passes only if it satisfies the criterion. Fluency, length and
  confidence are irrelevant.
- HONESTY IS NOT A PASS. When the gold says the question is answerable from the
  graph, an answer that declines - "the query returned nothing", "I cannot
  determine this", "you should check the portal" - FAILS, however well judged
  and well worded the refusal is. It is the right behaviour for a system that
  failed, and it is still a failed answer. Only award a pass for a refusal when
  the gold says the question is NOT answerable.
- For counting questions, the stated number must fall inside the stated
  tolerance. An answer that dodges the count does not pass.
- When an answer lists examples AND states a total, the total must match the
  gold count. Reporting the number of items it happened to list ("40
  publications") as though it were the total, when the gold count is far
  larger, is a failure: it misinforms the reader about the size of the result.
- For questions where the criterion says the graph records nothing for the
  relation asked about: an answer passes by saying so. It FAILS if it claims
  that relation holds anyway. But read the criterion closely — where it says a
  DIFFERENT relation is populated (publications describing an algorithm rather
  than using it), naming those is correct and better than a flat refusal, so
  long as the answer keeps the two relations distinct. Do not fail an answer
  for volunteering true, clearly-labelled neighbouring facts.
- Presenting general knowledge as MaRDI graph content is a failure. Clearly
  labelling it as outside knowledge is not.
- Invented identifiers (Qxxx that the answer cannot have retrieved) are a
  failure.

Reply with JSON only, no prose, no code fences:
{"a_pass": true/false, "a_reason": "<=20 words",
 "b_pass": true/false, "b_reason": "<=20 words",
 "better": "A" | "B" | "tie", "better_reason": "<=20 words"}
"""


def judge_one(judge: LLM, q: dict, ans_a: str, ans_b: str) -> dict:
    gold = q["gold"]
    facts = [f"criterion: {gold['criterion']}"]
    if "numeric_answer" in gold:
        tol = gold.get("numeric_tolerance_pct", 10)
        facts.append(f"true count: {gold['numeric_answer']} (tolerance +-{tol}%)")
    if gold.get("expected_count") is not None:
        facts.append(f"publications linked in graph: {gold['expected_count']}")
    if gold.get("sample_titles"):
        facts.append("some real linked titles: " + "; ".join(gold["sample_titles"][:4]))
    facts.append(f"answerable from the graph: {gold['answerable']}")

    prompt = (f"QUESTION: {q['question']}\n\nGOLD:\n" + "\n".join(f"- {f}" for f in facts)
              + f"\n\nANSWER A:\n{ans_a}\n\nANSWER B:\n{ans_b}\n")
    raw = judge.complete(JUDGE_SYSTEM, prompt, max_tokens=400)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"error": "unparseable judge reply", "raw": raw[:300]}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        return {"error": f"bad json: {exc}", "raw": raw[:300]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--judge-model", default=None, help="defaults to the main model")
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=17, help="controls A/B position shuffling")
    args = ap.parse_args()

    spec = json.loads(TEST_SET.read_text(encoding="utf-8"))
    questions = spec["questions"]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        questions = [q for q in questions if q["id"] in wanted]

    llm = LLM(provider=args.provider, model=args.model)
    judge = LLM(provider=args.provider, model=args.judge_model or args.model)
    print("Loading property catalog…")
    catalog = fetch_property_catalog()
    print(f"System: {llm.provider}/{llm.model}   Judge: {judge.provider}/{judge.model}   "
          f"Questions: {len(questions)}\n")

    rng = random.Random(args.seed)
    results: list[dict] = []

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['id']} {q['shape']:<22} "
              f"{q['question'][:52]}", flush=True)
        row: dict = {"id": q["id"], "category": q["category"], "shape": q["shape"],
                     "question": q["question"], "gold": q["gold"]}

        for name, fn in (("naive", lambda: answer_naive(q["question"], llm)),
                         ("full", lambda: route_and_answer(q["question"], llm, catalog))):
            t = time.time()
            try:
                trace = fn()
                row[f"{name}_answer"] = trace.answer
                row[f"{name}_route"] = trace.route
                row[f"{name}_error"] = None
            except Exception as exc:
                row[f"{name}_answer"] = ""
                row[f"{name}_route"] = None
                row[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
            row[f"{name}_latency_s"] = round(time.time() - t, 2)

        # Blind the judge to which system produced which answer, and shuffle the
        # position per question so a position bias cannot favour one branch.
        naive_is_a = rng.random() < 0.5
        a, b = (("naive", "full") if naive_is_a else ("full", "naive"))
        verdict = judge_one(judge, q, row[f"{a}_answer"], row[f"{b}_answer"])
        row["judge_raw"] = verdict
        row["naive_is_a"] = naive_is_a

        if "error" in verdict:
            row["naive_pass"] = row["full_pass"] = None
            row["better"] = None
            print(f"      JUDGE ERROR: {verdict['error']}")
        else:
            row[f"{a}_pass"] = bool(verdict.get("a_pass"))
            row[f"{b}_pass"] = bool(verdict.get("b_pass"))
            row[f"{a}_reason"] = verdict.get("a_reason", "")
            row[f"{b}_reason"] = verdict.get("b_reason", "")
            pick = verdict.get("better")
            row["better"] = ("tie" if pick == "tie"
                             else a if pick == "A" else b if pick == "B" else None)
            print(f"      naive={'PASS' if row['naive_pass'] else 'FAIL'}  "
                  f"full={'PASS' if row['full_pass'] else 'FAIL'}  "
                  f"better={row['better']}  [{row['full_route']}]")

        results.append(row)
        if args.out:
            _write(args.out, llm, judge, spec, results, partial=True)

    _summary(results)
    if args.out:
        _write(args.out, llm, judge, spec, results, partial=False)
        print(f"\nReport: {args.out}")
    return 0


def _write(out: str, llm, judge, spec: dict, results: list[dict], partial: bool) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({
        "model": f"{llm.provider}/{llm.model}",
        "judge_model": f"{judge.provider}/{judge.model}",
        "test_set_version": spec["version"],
        "grading": "llm_judge",
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
        "partial": partial,
        "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def _summary(results: list[dict]) -> None:
    scored = [r for r in results if r.get("naive_pass") is not None]
    if not scored:
        print("\nNo scored rows.")
        return
    n = len(scored)
    nv = sum(r["naive_pass"] for r in scored)
    fl = sum(r["full_pass"] for r in scored)
    print(f"\n{'=' * 64}\nnaive {nv}/{n}   full {fl}/{n}")

    by: dict[str, list[dict]] = {}
    for r in scored:
        by.setdefault(r["category"], []).append(r)
    print(f"\n  {'category':<14} {'naive':>8} {'full':>8}")
    for cat, rows in sorted(by.items()):
        print(f"  {cat:<14} {sum(r['naive_pass'] for r in rows):>4}/{len(rows):<3} "
              f"{sum(r['full_pass'] for r in rows):>4}/{len(rows):<3}")

    wins = sum(1 for r in scored if r["better"] == "full")
    loss = sum(1 for r in scored if r["better"] == "naive")
    ties = sum(1 for r in scored if r["better"] == "tie")
    print(f"\n  judge preference: full {wins} / naive {loss} / tie {ties}")

    regress = [r["id"] for r in scored if r["naive_pass"] and not r["full_pass"]]
    if regress:
        print(f"  naive passes where full fails: {', '.join(regress)}")


if __name__ == "__main__":
    raise SystemExit(main())
