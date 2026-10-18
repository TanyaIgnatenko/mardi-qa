"""Phase-1 harness: stage-level metrics over the six-question set.

Phase 1 answers "does the instrument work", not "which model wins". Six questions
cannot rank anything, but they can show that every metric computes, that the
trace carries what the metrics need, and where in the pipeline a failure sits.

Routing is switched off: the set is graph-only by design, and leaving the router
in would let a misroute confound every other measurement. We have seen that
happen — a router that silently sent 41 of 41 publication questions away from the
graph made the whole run unreadable.

    python eval/run_phase1.py --out eval/reports/phase1.json
    python eval/run_phase1.py --only P1,P4 --out eval/reports/smoke.json
    python eval/run_phase1.py --regrade eval/reports/phase1.json   # metrics only, no API
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# Must precede the mardi_qa import: config reads the environment at import time.
os.environ.setdefault("ROUTER_ENABLED", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

from mardi_qa import config
from mardi_qa.entity_linking import fetch_property_catalog
from mardi_qa.llm import LLM
from mardi_qa.pipeline import answer_question
from mardi_qa.sparql_client import run_query

import stage_metrics as sm

SETS = {
    "phase1": "test_set_phase1.json",   # hard set: localises new defects
    "easy": "test_set_easy.json",       # known-good set: catches regressions
}

JUDGE_SYSTEM = """\
You grade one answer from a knowledge-graph question-answering system. Two
judgments, both against the gold statement supplied.

1. fact_asserted — does the answer ASSERT the gold fact as true? Judge ONLY
   that. Extra material is not your concern: an answer that states the gold fact
   and then adds things the question did not ask for still PASSES here, because
   the surplus is counted separately and exactly, by metrics that do not need an
   opinion. Asked which algorithms suit a sparse SPD system, an answer naming
   both gold algorithms correctly plus six more was failed on this line, which
   hid the fact that the answer the user wanted was in there.
   PASSES: the fact stated plainly, including when stated as what the graph or
   the query returned. "The query returned 1438 publications in MaRDI" asserts
   1438 — grounding a fact in the data is required of this system, not a defect,
   and saying where a number came from is never a reason to fail.
   FAILS: the fact denied ("never use this method here" names the method and
   still fails); withdrawn after being stated ("0 results, but that does not
   mean there are none"); or credited to someone outside the data, such as
   another author or the reader's own judgement.
   When the gold says the graph records nothing for the relation asked about, an
   answer passes by saying so plainly and fails by casting doubt on its own zero.
2. scope_limited — does the answer limit its completeness to what the graph
   records, rather than implying it is complete for the field? Judge the meaning,
   not the presence of any particular phrase.

Also rate the query, if one is shown:
3. query_sound — is this a reasonable query for the question: right direction,
   no needless joins, not so narrow that it excludes valid matches?

Reply with JSON only:
{"fact_asserted": true/false, "fact_reason": "<=20 words",
 "scope_limited": true/false,
 "query_sound": true/false/null, "query_reason": "<=20 words"}
"""


def gold_count(gold_sparql: str | None) -> int | None:
    """The number the gold query returns right now.

    Counts are compared exactly, and that is only sound because the reference is
    re-measured at run time against the same graph the system queries. A gold
    number frozen at build time would drift as the graph grows, which is what the
    10% tolerance used to absorb — and that tolerance also waved through wrong
    queries: a model applying one filter too many returns 1400 against 1438 and
    passes. A correct query cannot be approximately right.
    """
    if not gold_sparql:
        return None
    try:
        rows = run_query(f"{config.SPARQL_PREFIXES}\n{gold_sparql}").rows
    except Exception:
        return None
    return sm._first_number(rows)


def gold_row_qids(gold_sparql: str | None) -> set[str] | None:
    """Execute the gold query once and keep the identifiers it returns.

    This is the reference for execution accuracy: a generated query is right when
    it returns the same set, whatever its text looks like.
    """
    if not gold_sparql:
        return None
    try:
        result = run_query(f"{config.SPARQL_PREFIXES}\n{gold_sparql}")
    except Exception as exc:
        print(f"      gold query failed: {type(exc).__name__}: {exc}")
        return None
    return sm.qids_in_rows(result.rows)


def judge_answer(judge: LLM, q: dict, answer: str, query: str) -> dict:
    prompt = (f"QUESTION: {q['question']}\n\n"
              f"GOLD STATEMENT:\n{q['gold_fact']}\n\n"
              f"ANSWER:\n{answer}\n\n"
              f"QUERY THE SYSTEM RAN:\n{query or '(none)'}\n")
    raw = judge.complete(JUDGE_SYSTEM, prompt, max_tokens=350)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"error": "unparseable judge reply"}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        return {"error": f"bad json: {exc}"}


def graph_labels(answer: str) -> dict[str, str]:
    """The graph's own label for every identifier the answer cites.

    Read from the graph rather than from the trace on purpose: the naming check
    exists for the case where the query never asked for labels, so the trace is
    exactly where they are missing.
    """
    return dict(_labels_for(tuple(sorted(set(sm.QID.findall(answer or ""))))))


@lru_cache(maxsize=256)
def _labels_for(qids: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """Cached, because a regrade re-asks the same question of the same graph and
    this endpoint answers an excess of requests with a 429 that fails the run."""
    if not qids:
        return ()
    query = (f"{config.SPARQL_PREFIXES}\nSELECT ?x ?xLabel WHERE {{ VALUES ?x {{ "
             + " ".join("wd:" + q for q in qids)
             + ' } SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } }')
    try:
        rows = run_query(query).rows
    except Exception:
        return ()
    return tuple((r.get("x", "").rsplit("/", 1)[-1], r.get("xLabel", "")) for r in rows)


def gold_label_map(q: dict) -> dict[str, str]:
    """{label: qid} for the entities this question expects, so a result that came
    back as names can still be credited with having found them."""
    required = tuple(sorted(q.get("required_entities") or ()))
    return {" ".join(label.split()).lower(): qid
            for qid, label in _labels_for(required) if label and label != qid}


def measure(q: dict, trace, gold_rows: set[str] | None) -> dict:
    """Every deterministic metric for one question, from the trace alone."""
    linked = [e.qid for e in trace.linked_entities]
    retrieved = set()
    for r in trace.results:
        retrieved |= sm.qids_in_rows(r.rows)
    # The schema joins the rows here: a quoted property label is evidence the
    # model was genuinely shown, and checking quotes against the result rows
    # alone flagged "implements algorithm" as an unsupported citation.
    context_text = "\n".join(
        " | ".join(str(v) for v in row.values()) for row in trace.context_rows
    ) + "\n" + (trace.fallback_context or "") + "\n" + getattr(trace, "schema_text", "") \
        + "\n" + getattr(trace, "identifier_gloss", "")

    return {
        "entities": sm.entity_metrics(trace.extracted, linked, q),
        "sparql": sm.sparql_metrics(trace.queries, trace.results, gold_rows, q),
        "retrieval": sm.retrieval_metrics(trace.results, trace.context_rows, q,
                                          gold_label_map(q)),
        "schema": sm.schema_metrics(trace.properties_shown, q),
        "sources": sm.source_metrics(trace.answer, retrieved, q, linked),
        "quotes": sm.quote_fidelity(trace.answer, context_text),
        "naming": sm.naming_metrics(trace.answer, graph_labels(trace.answer)),
        "structure": sm.structural_validity(trace.extracted, trace.queries,
                                            trace.query_errors),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="phase1", choices=sorted(SETS),
                    help="phase1 = the hard diagnostic set; easy = five questions "
                         "the pipeline already answers, used as a regression guard")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--provider", default=None,
                    help="anthropic or ollama — the system under test "
                         "(default: config.LLM_PROVIDER)")
    ap.add_argument("--judge-provider", default="anthropic",
                    help="backend for the judge; stays anthropic regardless of "
                         "which backend is under test")
    ap.add_argument("--judge-model", default=None,
                    help="should NOT be the model under test — a judge that is "
                         "the same model grades its own output")
    ap.add_argument("--regrade", default=None, metavar="REPORT",
                    help="recompute metrics from a saved report; no API calls")
    args = ap.parse_args()

    test_set = Path(__file__).parent / SETS[args.set]
    args.out = args.out or f"eval/reports/{args.set}.json"
    spec = json.loads(test_set.read_text(encoding="utf-8"))
    questions = spec["questions"]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        questions = [q for q in questions if q["id"] in wanted]

    if args.regrade:
        return _regrade(args.regrade, {q["id"]: q for q in spec["questions"]})

    print(f"Set: {args.set} ({len(spec['questions'])} questions) — {spec['purpose'][:72]}…")

    llm = LLM(provider=args.provider, model=args.model)
    # The judge stays on Anthropic even when a local model is under test: a judge
    # must not be the system it grades, and the local candidates are exactly what
    # is being compared.
    # Provider pinned, not inherited: the judge followed config.LLM_PROVIDER until
    # that default became ollama, at which point it tried to reach a Claude model
    # on the ZIB server and the run died on a 403. The grader must not move when
    # the system under test moves.
    judge = LLM(provider=args.judge_provider, model=args.judge_model)
    if judge.model == llm.model and judge.provider == llm.provider:
        print(f"WARNING: judge and system are both {judge.model}. A model grading "
              f"its own output favours it; use --judge-model for a comparison.\n")

    # Load the model before the clock starts on question one. The server evicts
    # idle models, and a reload costs minutes: qwen3.5:35b spent 153 seconds on E1
    # in one run against 13 in another, which inflates its reported latency and,
    # on the longer sets, can exhaust the run's time budget outright.
    warm_up_s = None
    if llm.provider == "ollama":
        print("Warming up the model…", end=" ", flush=True)
        started = time.time()
        for attempt in range(1, 4):
            try:
                llm.complete("Reply with the single word OK.", "Say OK", max_tokens=8)
                warm_up_s = round(time.time() - started, 1)
                print(f"ready in {warm_up_s:.0f}s", flush=True)
                break
            except Exception as exc:
                print(f"[{attempt}]", end=" ", flush=True)
                if attempt == 3:
                    print(f"never became ready — {type(exc).__name__}", flush=True)

    print("Loading property catalog…")
    catalog = fetch_property_catalog()
    print(f"System: {llm.provider}/{llm.model}   Judge: {judge.model}   "
          f"Router: {'off' if not config.ROUTER_ENABLED else 'ON (unexpected)'}   "
          f"Questions: {len(questions)}\n")

    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['id']}  {q['question'][:58]}", flush=True)
        gold_rows = gold_row_qids(q.get("gold_sparql"))
        if gold_rows is not None:
            print(f"      gold query returns {len(gold_rows)} identifiers")
        # Re-measure the reference count so the comparison can be exact.
        if (q.get("answer_shape") or "") == "count":
            live = gold_count(q.get("gold_sparql"))
            if live is not None:
                if live != q.get("numeric_answer"):
                    print(f"      gold count drifted: stored {q.get('numeric_answer')} "
                          f"-> live {live}; using the live value")
                q = {**q, "numeric_answer": live, "numeric_tolerance_pct": 0}

        started = time.time()
        try:
            trace = answer_question(q["question"], llm, catalog)
            err = None
        except Exception as exc:
            trace, err = None, f"{type(exc).__name__}: {exc}"

        row: dict = {"id": q["id"], "question": q["question"],
                     "latency_s": round(time.time() - started, 2), "error": err}
        if trace is not None:
            row["metrics"] = measure(q, trace, gold_rows)
            row["judge"] = judge_answer(judge, q, trace.answer,
                                        trace.queries[-1] if trace.queries else "")
            row["trace"] = {
                "extracted": trace.extracted,
                "linked": [{"qid": e.qid, "label": e.label} for e in trace.linked_entities],
                "queries": trace.queries,
                # Raw rows, not just their count: metric definitions are still
                # moving, and storing the rows is what lets a redefinition be
                # applied to past runs instead of re-running the system.
                "query_results": [r.rows for r in trace.results],
                "context_rows": trace.context_rows,
                "properties_shown": trace.properties_shown,
                "schema_text": trace.schema_text,
                "identifier_gloss": trace.identifier_gloss,
                "query_errors": trace.query_errors,
                "used_fallback": trace.used_fallback,
                "fallback_context": trace.fallback_context[:4000],
                "gold_rows": sorted(gold_rows) if gold_rows is not None else None,
            }
            row["answer"] = trace.answer
            _print_row(row)
        else:
            print(f"      FAILED: {err}")

        results.append(row)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"model": f"{llm.provider}/{llm.model}", "judge_model": judge.model,
             "test_set_version": spec["version"],
             "router": "disabled",
             "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
             "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")

    _summary(results)
    print(f"\nReport: {args.out}")
    return 0


def _print_row(row: dict) -> None:
    m, j = row["metrics"], row.get("judge", {})
    e, s, r, src = m["entities"], m["sparql"], m["retrieval"], m["sources"]
    print(f"      linking recall={_f(e['linking_recall'])} precision={_f(e['linking_precision'])}"
          f"  | query exact first={s['first_query_exact']} final={s['final_query_exact']}"
          f" attempts={s['attempts']}")
    print(f"      retrieval={_f(r['retrieval_recall'])} context={_f(r['context_recall'])}"
          f" loss={_f(r['truncation_loss'])}"
          f"  | sources recall={_f(src['source_recall'])} precision={_f(src['source_precision'])}")
    flags = []
    if src["fabricated"]:
        flags.append(f"FABRICATED {src['fabricated']}")
    if m["quotes"]["unsupported_quotes"]:
        flags.append(f"UNSUPPORTED QUOTE x{len(m['quotes']['unsupported_quotes'])}")
    if src["superfluous"]:
        flags.append(f"superfluous x{len(src['superfluous'])}")
    print(f"      judge: fact={j.get('fact_asserted')} scope={j.get('scope_limited')}"
          f" query_sound={j.get('query_sound')}"
          + (f"   {' / '.join(flags)}" if flags else ""))


def _f(x) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def _yn(x) -> str:
    return "n/a" if x is None else ("yes" if x else "NO")


def _count(xs) -> str:
    return "n/a" if xs is None else str(len(xs))


def _frac(got: int, total: int) -> str:
    return "n/a" if not total else f"{got}/{total}"


def _extra(sources: dict) -> str:
    """Everything cited that is not part of the answer, surplus or excluded."""
    surplus, wrong = sources["superfluous"], sources["wrongly_cited"]
    if surplus is None and not wrong:
        return "n/a"  # the gold does not say what the answer should contain
    return str(len(surplus or ()) + len(wrong or ()))


def _summary(results: list[dict]) -> None:
    scored = [r for r in results if "metrics" in r]
    if not scored:
        print("\nNothing scored.")
        return
    # Transposed, because the stage names are the point: a metric only helps if
    # the reader can tell which stage it accuses without consulting a legend.
    # Questions are few and their ids are short, so they go across the top.
    rows = [
        ("Found the right entities",      lambda m, j: _f(m["entities"]["linking_recall"])),
        ("Shown the needed properties",   lambda m, j: _f(m["schema"]["schema_coverage"])),
        ("Built the right query",         lambda m, j: _yn(m["sparql"]["final_query_exact"])),
        ("Query needed a repair",         lambda m, j: _yn(bool(m["sparql"]["repair_fired"]))),
        ("Retrieved the right rows",      lambda m, j: _f(m["retrieval"]["retrieval_recall"])),
        ("Rows survived into context",    lambda m, j: _f(m["retrieval"]["context_recall"])),
        ("Number reached the prompt",     lambda m, j: _yn(m["retrieval"]["count_in_context"])),
        ("Lost to truncation",            lambda m, j: _f(m["retrieval"]["truncation_loss"])),
        # "Cited" read as quotation; this row is about portal links, and the one
        # below it about quoted text, so they say which they mean.
        ("Gave the needed links",         lambda m, j: _frac(m["sources"]["linked_correct"],
                                                            m["sources"]["linked_required"])),
        # Counted, not just scored. P1 cited both algorithms the gold asks for and
        # six category nodes besides, and a table of recalls and verdicts showed
        # that as failure throughout — "nothing missing, six extra" is a different
        # state from "missed everything", and the fix for each is different too.
        ("Missing facts from the answer", lambda m, j: _count(m["sources"]["missed"])),
        # Surplus and wrongly-cited together, because the row answers one
        # question — how much of this was not the answer. Splitting them hid the
        # six category nodes in P1: naming them in the gold's excluded list moved
        # them out of `superfluous` and the row dropped to zero while the noise
        # was still there. The JSON keeps the two apart for diagnosis.
        ("Extra facts, not an answer",    lambda m, j: _extra(m["sources"])),
        ("Links came from the data",      lambda m, j: _f(m["sources"]["source_precision"])),
        ("Invented identifiers",          lambda m, j: str(len(m["sources"]["fabricated"]))),
        ("Called an item by a wrong name", lambda m, j: str(len(m["naming"]["mislabelled"]))),
        ("Quotes unsupported by data",    lambda m, j: str(len(m["quotes"]["unsupported_quotes"]))),
        ("Asserted the known fact",       lambda m, j: _yn(j.get("fact_asserted"))),
    ]
    width = max(len(label) for label, _ in rows) + 2
    print(f"\n{'=' * (width + 7 * len(scored))}")
    print(" " * width + "".join(f"{r['id']:>7}" for r in scored))
    for label, get in rows:
        cells = "".join(f"{get(r['metrics'], r.get('judge') or {}):>7}" for r in scored)
        print(f"{label:<{width}}{cells}")

    fab = sum(len(r["metrics"]["sources"]["fabricated"]) for r in scored)
    bad_quotes = sum(len(r["metrics"]["quotes"]["unsupported_quotes"]) for r in scored)
    repairs = sum(r["metrics"]["sparql"]["repair_fired"] for r in scored)
    truncated = sum(bool(r["metrics"]["retrieval"].get("limit_truncated")) for r in scored)
    print(f"\n  fabricated identifiers: {fab}   unsupported quotes: {bad_quotes}")
    print(f"  repair fired: {repairs}/{len(scored)}   capped by LIMIT: {truncated}/{len(scored)}")
    print("\n  Phase 1 checks the instrument: every column above should contain a "
          "number or a deliberate n/a.\n  An all-n/a column means that metric never "
          "fired and the set does not exercise it.")


class _StoredResult:
    """Minimal stand-in for SparqlResult so the metrics run on stored rows."""

    def __init__(self, rows: list[dict]):
        self.rows = rows


def _regrade(path: str, by_id: dict) -> int:
    """Recompute every deterministic metric from a saved report.

    No system run and no API call: the judge's two verdicts are carried over
    unchanged, everything else is recomputed. This is what makes a metric
    definition cheap to change — redefine it, regrade every past run, compare.
    """
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    print(f"Regrading {path} — {len(report['results'])} rows, no API calls.\n")

    for row in report["results"]:
        trace_data = row.get("trace")
        if not trace_data:
            continue
        q = by_id[row["id"]]

        class _T:
            extracted = trace_data["extracted"]
            linked_entities = [type("E", (), {"qid": e["qid"]})() for e in trace_data["linked"]]
            queries = trace_data["queries"]
            results = [_StoredResult(rows) for rows in trace_data["query_results"]]
            context_rows = trace_data["context_rows"]
            query_errors = trace_data["query_errors"]
            used_fallback = trace_data["used_fallback"]
            fallback_context = trace_data.get("fallback_context", "")
            answer = row.get("answer", "")
            # Absent from reports written before the schema metric existed, so
            # regrading one of those yields no coverage rather than crashing.
            properties_shown = trace_data.get("properties_shown", [])
            schema_text = trace_data.get("schema_text", "")
            identifier_gloss = trace_data.get("identifier_gloss", "")

        gold = trace_data.get("gold_rows")
        before = row.get("metrics", {})
        row["metrics"] = measure(q, _T, set(gold) if gold is not None else None)
        _report_delta(row["id"], before, row["metrics"])

    out = Path(path).with_name(Path(path).stem + "_regraded.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _summary(report["results"])
    print(f"\nRegraded report: {out}")
    return 0


def _report_delta(qid: str, before: dict, after: dict) -> None:
    """Name the metrics a redefinition actually moved, so a silent change is loud."""
    changed = []
    for group in after:
        for key, new in after[group].items():
            old = (before.get(group) or {}).get(key, "<absent>")
            if old != new:
                changed.append(f"{group}.{key}: {old} -> {new}")
    print(f"  {qid}: {'; '.join(changed) if changed else 'unchanged'}")


if __name__ == "__main__":
    raise SystemExit(main())
