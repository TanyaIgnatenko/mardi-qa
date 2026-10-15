"""Self-test for the regression grader.

Guards the gold checks themselves: a check must fail a bad answer AND pass a
legitimate one. Several naive checks were rejected during design because they
punished correct answers (e.g. banning the phrase "will maintain" also fails an
honest refusal that says the graph cannot know who will maintain the package).

Run:  python eval/test_grading.py      (no API key or network needed)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_regression import grade  # noqa: E402

SPEC = json.loads((Path(__file__).parent / "test_set.json").read_text(encoding="utf-8"))
BY_ID = {q["id"]: q for q in SPEC["questions"]}

# The genuine defective answer produced by the 2026-06-23 baseline run: the
# pipeline queried P342, saw only "R", and told the user the data was missing -
# while the real imports sit in P585.
REAL_DEFECT = (
    "Based on the retrieved knowledge-graph data, **dplyr** (Q25186) depends on **R** (Q13519). "
    "The query searched for dependencies of dplyr using the dependency property (P342), and only "
    "one dependency was returned: the R programming language. However, this appears to be "
    "incomplete information. The R software package dplyr typically depends on several other R "
    "packages (such as tibble, magrittr, rlang, etc.), but these are not captured in the "
    "retrieved data."
)
FIXED = (
    "dplyr (Q25186) imports methods, utils, magrittr, tibble (Q31187), rlang (Q54216), R6, cli, "
    "lifecycle, pillar, tidyselect and vctrs."
)

# (question id, answer, linked qids, expected verdict, label)
CASES = [
    ("D01", REAL_DEFECT, {"Q25186", "Q13519"}, False, "known defect is caught"),
    ("D01", FIXED, {"Q25186", "Q31187", "Q54216"}, True, "fixed pipeline passes"),
    ("D02", REAL_DEFECT, {"Q25186", "Q13519"}, False, "false 'not captured' claim caught"),

    ("H01", "data.table extends data.frame; dplyr is a grammar of data manipulation. "
            "No comparison is recorded.", {"Q33745", "Q25186"}, True, "descriptive contrast ok"),
    ("H01", "data.table is 42,000 times faster on 1,000,000 rows.",
     {"Q33745", "Q25186"}, False, "invented benchmark caught"),

    ("H02", "Neither is better; the graph does not record comparisons.", set(), True,
     "honest answer may still use the word 'better'"),
    ("H02", "data.table is better for large datasets.", set(), False, "bare opinion caught"),

    ("H03", "The graph cannot say who will maintain ggplot2 in 2030; currently it is "
            "Thomas Lin Pedersen.", set(), True, "honest refusal may use 'will maintain'"),
    ("H03", "Hadley Wickham will maintain it.", set(), False, "confident fabrication caught"),

    ("H04", "dplyr was downloaded 1,204,553 times last month.", set(), False,
     "fabricated statistic caught"),
    ("H04", "The knowledge graph does not record download statistics.", set(), True,
     "honest decline passes"),

    ("G01", "There are 20,321 R packages.", {"Q57080"}, True, "exact count"),
    ("G01", "There are 19,500 R packages.", {"Q57080"}, True, "within 10% tolerance"),
    ("G01", "There are 500 R packages.", {"Q57080"}, False, "far off is caught"),
]


def main() -> int:
    failures = 0
    for qid, answer, linked, expect, label in CASES:
        got = grade(BY_ID[qid], answer, linked)["passed"]
        if got != expect:
            failures += 1
            print(f"BUG  {qid}: expected {'PASS' if expect else 'FAIL'}, "
                  f"got {'PASS' if got else 'FAIL'}  [{label}]")
        else:
            print(f"ok   {qid}  [{label}]")

    print()
    if failures:
        print(f"{failures} grading test(s) failed")
        return 1
    print(f"all {len(CASES)} grading tests pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
