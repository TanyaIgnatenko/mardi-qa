"""Does the graph route need a semantic entry point?

Matched pairs: every problem class is asked twice, once in the graph's own words
and once paraphrased so no distinctive content word is shared with the stored
label or description. Only the phrasing varies, so the difference between the two
isolates exactly what a lexical lookup can and cannot reach.

    lexical passes, paraphrase fails  -> embeddings would earn their place
    both pass                         -> the existing full-text fallback suffices
    both fail                         -> the problem is upstream of retrieval

    python eval/run_exploratory.py --out eval/reports/exploratory.json
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

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

from mardi_qa.entity_linking import fetch_property_catalog
from mardi_qa.gateway import route_and_answer
from mardi_qa.llm import LLM

TEST_SET = Path(__file__).parent / "test_set_exploratory.json"

JUDGE_SYSTEM = """\
You grade one answer from a question-answering system over the MaRDI knowledge
graph, against a criterion naming the algorithms the graph actually links to the
problem in the question.

Pass only if the answer names at least one of those algorithms AND presents it as
coming from the graph. Fail if it answers from general knowledge without reaching
the graph, if it declines, or if it names algorithms the criterion does not list
and none that it does. General-knowledge answers can be entirely correct
mathematics and still fail here: the question is whether the graph was reached.

Reply with JSON only:
{"pass": true/false, "reason": "<=20 words", "from_graph": true/false}
"""


def judge(judge_llm: LLM, q: dict, answer: str) -> dict:
    prompt = (f"QUESTION: {q['question']}\n\nCRITERION: {q['criterion']}\n"
              f"Algorithms the graph links: {', '.join(q['gold_algorithms'])}\n\n"
              f"ANSWER:\n{answer}\n")
    raw = judge_llm.complete(JUDGE_SYSTEM, prompt, max_tokens=300)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"error": "unparseable judge reply"}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        return {"error": str(exc)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/reports/exploratory.json")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    spec = json.loads(TEST_SET.read_text(encoding="utf-8"))
    questions = spec["questions"]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        questions = [q for q in questions if q["id"] in wanted]

    llm = LLM()
    print("Loading property catalog…")
    catalog = fetch_property_catalog()
    print(f"Model: {llm.provider}/{llm.model}   Questions: {len(questions)}\n")

    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['id']} ({q['variant']:<10}) "
              f"{q['question'][:56]}", flush=True)
        started = time.time()
        try:
            trace = route_and_answer(q["question"], llm, catalog)
            answer, route = trace.answer, trace.route
            linked = [e.qid for e in trace.linked_entities]
            err = None
        except Exception as exc:
            answer, route, linked, err = "", None, [], f"{type(exc).__name__}: {exc}"

        verdict = judge(llm, q, answer) if answer else {"pass": False, "reason": "no answer"}
        row = {**{k: q[k] for k in ("id", "pair", "variant", "problem_qid", "question")},
               "route": route, "linked_qids": linked,
               "found_problem_class": q["problem_qid"] in linked,
               "passed": bool(verdict.get("pass")),
               "from_graph": bool(verdict.get("from_graph")),
               "reason": verdict.get("reason", ""), "error": err,
               "latency_s": round(time.time() - started, 2), "answer": answer}
        results.append(row)
        print(f"      {'PASS' if row['passed'] else 'FAIL'}  route={route}  "
              f"linked={linked or '-'}  from_graph={row['from_graph']}")

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"model": f"{llm.provider}/{llm.model}",
             "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
             "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")

    _summary(results)
    print(f"\nReport: {args.out}")
    return 0


def _summary(results: list[dict]) -> None:
    lex = [r for r in results if r["variant"] == "lexical"]
    par = [r for r in results if r["variant"] == "paraphrase"]
    print(f"\n{'=' * 66}")
    for name, rows in (("lexical   ", lex), ("paraphrase", par)):
        if rows:
            print(f"  {name}  passed {sum(r['passed'] for r in rows)}/{len(rows)}   "
                  f"reached the graph {sum(r['from_graph'] for r in rows)}/{len(rows)}   "
                  f"linked the problem class {sum(r['found_problem_class'] for r in rows)}/{len(rows)}")

    by_pair: dict[str, dict] = {}
    for r in results:
        by_pair.setdefault(r["pair"], {})[r["variant"]] = r
    gap = [p for p, v in by_pair.items()
           if v.get("lexical", {}).get("passed") and not v.get("paraphrase", {}).get("passed")]
    both = [p for p, v in by_pair.items()
            if v.get("lexical", {}).get("passed") and v.get("paraphrase", {}).get("passed")]
    neither = [p for p, v in by_pair.items()
               if not v.get("lexical", {}).get("passed")
               and not v.get("paraphrase", {}).get("passed")]
    print(f"\n  lexical only (embeddings would help): {len(gap)}  {', '.join(sorted(gap))}")
    print(f"  both variants pass (they would not):  {len(both)}  {', '.join(sorted(both))}")
    print(f"  neither passes (problem is upstream): {len(neither)}  {', '.join(sorted(neither))}")


if __name__ == "__main__":
    raise SystemExit(main())
