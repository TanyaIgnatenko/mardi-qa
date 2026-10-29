"""Convert the 50-question set into the shape the stage harness grades.

test_set_v2.json was written for a pass/fail judge: its gold says what a correct
answer must contain, not which query produces it, so the stage metrics had
nothing to compare against and reported n/a down the column. Everything they
need is derivable, because every question carries a subject_qid and one of seven
shapes, and the relation each shape uses is one of the directions verified
against this endpoint.

    python eval/build_test_set_staged.py

Writes test_set_v2_staged.json. The gold queries run once here, to record which
entities a list answer must contain; the harness re-runs them on every
evaluation, so the expected result never goes stale.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mardi_qa import config  # noqa: E402
from mardi_qa.sparql_client import run_query, SparqlError  # noqa: E402

QID_RE = re.compile(r"Q\d{2,9}")

LABEL = 'SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }'

# Above a list this long, naming every member is not what a good answer does, so
# the question is scored as a sample of a known total rather than a set to match.
TRUNCATE_ABOVE = 50


def gold_query(q: dict) -> str | None:
    """The query that produces this question's answer, by shape."""
    g, subj = q["gold"], q["gold"]["subject_qid"]
    shape = q["shape"]

    if shape == "reverse_traversal":
        return f"SELECT ?p ?pLabel WHERE {{ ?p wdt:P1463 wd:{subj} . {LABEL} }}"
    if shape == "aggregation":
        return f"SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {{ ?p wdt:P1463 wd:{subj} }}"
    if shape == "two_way_join":
        return (f"SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {{ "
                f"?p wdt:P1463 wd:{subj} . ?p wdt:P1463 wd:{g['second_qid']} }}")
    if shape == "join_with_date_filter":
        return (f"SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {{ "
                f"?p wdt:P1463 wd:{subj} ; wdt:P28 ?d . "
                f"FILTER(YEAR(?d) >= {g['since_year']}) }}")
    if shape == "algorithm_to_publications":
        # described by source: the algorithm is the subject, the paper the object.
        return f"SELECT ?p ?pLabel WHERE {{ wd:{subj} wdt:P286 ?p . {LABEL} }}"
    if shape == "algorithm_to_software":
        return f"SELECT ?s ?sLabel WHERE {{ wd:{subj} wdt:P1766 ?s . {LABEL} }}"
    if shape == "unanswerable":
        # The query a correct answer is entitled to run: it returns zero, and the
        # gold is that the zero must not become a claim that publications exist.
        return f"SELECT (COUNT(DISTINCT ?p) AS ?n) WHERE {{ ?p wdt:P1463 wd:{subj} }}"
    return None


COUNT_SHAPES = {"aggregation", "two_way_join", "join_with_date_filter", "unanswerable"}


def main() -> int:
    src = Path(__file__).parent / "test_set_v2.json"
    spec = json.loads(src.read_text(encoding="utf-8"))

    out, failures = [], 0
    for q in spec["questions"]:
        g = q["gold"]
        query = gold_query(q)
        if not query:
            print(f"  {q['id']}: shape {q['shape']!r} not handled — skipped")
            continue

        row = {
            "id": q["id"],
            "intent": q["category"],
            "question": q["question"],
            "gold_sparql": query,
            # The entity the question is about, which is what linking is graded
            # on — not the entities the answer consists of.
            "pivot_entities": [g["subject_qid"]],
            "gold_fact": g.get("criterion", ""),
        }

        if q["shape"] in COUNT_SHAPES:
            row["answer_shape"] = "count"
            row["numeric_answer"] = g.get("numeric_answer", g.get("expected_count", 0))
        else:
            total = g.get("expected_count")
            if total is not None and total > TRUNCATE_ABOVE:
                row["answer_shape"] = "truncated_list"
                row["expected_counts"] = {"publications": total}
            else:
                row["answer_shape"] = "list"
                try:
                    rows = run_query(f"{config.SPARQL_PREFIXES}\n{query}").rows
                except SparqlError as exc:
                    failures += 1
                    print(f"  {q['id']}: gold query failed — {str(exc)[:60]}")
                    rows = []
                # Both forms: this client hands back a bare "Q6825348" for some
                # queries and a full entity URI for others. Matching only the URI
                # silently produced nine questions with no required entities,
                # including one whose query demonstrably returns eleven rows.
                qids = sorted({
                    v.rsplit("/", 1)[-1] for r in rows for v in r.values()
                    if isinstance(v, str) and QID_RE.fullmatch(v.rsplit("/", 1)[-1])
                })
                row["required_entities"] = qids
                print(f"  {q['id']}: {len(qids)} required entities")

        out.append(row)

    dst = Path(__file__).parent / "test_set_v2_staged.json"
    dst.write_text(json.dumps({
        "version": "v2-staged-1.0",
        "purpose": ("The 50-question set, rewritten so the stage metrics apply: each "
                    "question's gold query is derived from its shape and subject, and "
                    "the expected result is re-measured against the graph on every run."),
        "questions": out,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    shapes = {}
    for q in spec["questions"]:
        shapes[q["shape"]] = shapes.get(q["shape"], 0) + 1
    print(f"\n{len(out)} of {len(spec['questions'])} questions converted"
          + (f", {failures} gold queries failed" if failures else ""))
    print(f"Written: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
