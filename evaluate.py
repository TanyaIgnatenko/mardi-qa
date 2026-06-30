"""Evaluate the MaRDI QA pipeline across multiple LLM backends.

Runs the same set of questions through several models, collects structural
metrics from PipelineTrace, asks Claude to judge each non-baseline answer
against the Claude baseline, and writes everything to eval_results/runs/<ts>/.

Usage:
    # Required environment:
    #   ANTHROPIC_API_KEY=sk-ant-...          (for Claude baseline + judge)
    #   OLLAMA_API_KEY=...                    (for ZIB Ollama models)
    # Optional:
    #   MARDI_SPARQL_ENDPOINT=...             (skip endpoint autodiscovery)
    #
    # Then:
    python evaluate.py

Edit CANDIDATES and QUESTIONS below to customize a run. Everything else is
configured through environment variables already used by the app.
"""

from __future__ import annotations

import json
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

from mardi_qa import config
from mardi_qa.entity_linking import fetch_property_catalog
from mardi_qa.llm import LLM
from mardi_qa.pipeline import PipelineTrace, answer_question


# ============================================================
# Configuration — edit this block to change what gets evaluated
# ============================================================

# (provider, model) pairs. The first entry is treated as the BASELINE that
# everyone else is compared against. Keep Claude first.
CANDIDATES: list[tuple[str, str]] = [
    ("anthropic", "claude-sonnet-4-5"),       # baseline (always first)
    ("ollama",    "gemma3:27b"),               # team default
    ("ollama",    "gemma4:31b"),               # next-gen Gemma — compare against gemma3
    ("ollama",    "qwen3-coder-next:latest"),  # code-specialized candidate
    ("ollama",    "qwen3.5:35b"),              # general-purpose Qwen — compare against coder
    ("ollama",    "phi4:14b"),                 # compact fast candidate
]

QUESTIONS: list[str] = [
    "What is the purpose of the CRAN package ggplot2?",
    "Which CRAN packages are related to linear regression?",
    "Which packages does dplyr depend on?",
    "Who maintains the CRAN package data.table?",
    "What is the difference between the CRAN packages data.table and dplyr?",
]

# Model used to judge candidate answers against the baseline answer.
# We keep this as the strongest Claude available.
JUDGE_PROVIDER = "anthropic"
JUDGE_MODEL = "claude-sonnet-4-5"

OUTPUT_ROOT = Path("eval_results")


# ============================================================
# MaRDI item validity — batch lookup against the portal's wbgetentities API
# ============================================================

import requests  # noqa: E402

# Process-wide cache: QID -> {"exists": bool, "label": Optional[str]}
_qid_info_cache: dict[str, dict] = {}


def _wbgetentities_batch(qids: list[str], batch_size: int = 50) -> None:
    """Populate _qid_info_cache for the given QIDs (existence + label).

    Uses the MediaWiki wbgetentities action API of the MaRDI portal.
    Skips QIDs already cached. Tolerates network errors silently.
    """
    todo = [q for q in qids if q not in _qid_info_cache]
    if not todo:
        return

    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        try:
            resp = requests.get(
                config.MEDIAWIKI_API,
                params={
                    "action": "wbgetentities",
                    "ids": "|".join(chunk),
                    "props": "labels",
                    "languages": "en",
                    "format": "json",
                },
                headers={"User-Agent": config.USER_AGENT},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json().get("entities", {})
            for qid in chunk:
                entry = data.get(qid)
                # wbgetentities marks missing items with "missing" key
                if not entry or "missing" in entry:
                    _qid_info_cache[qid] = {"exists": False, "label": None}
                else:
                    label = (
                        entry.get("labels", {}).get("en", {}).get("value")
                        if isinstance(entry.get("labels"), dict)
                        else None
                    )
                    _qid_info_cache[qid] = {"exists": True, "label": label}
        except Exception as exc:
            # On failure, mark these as "unknown" rather than missing — we use
            # exists=None to signal "could not verify".
            print(f"[warn] wbgetentities batch failed for {len(chunk)} QIDs: {exc}")
            for qid in chunk:
                _qid_info_cache.setdefault(qid, {"exists": None, "label": None})


def _labels_from_trace(trace: PipelineTrace) -> dict[str, str]:
    """Extract any QID->label mappings already present in the pipeline trace.

    Sources: linked_entities + SPARQL result rows containing ?itemLabel/?xLabel etc.
    These are free (no extra API call) and authoritative.
    """
    out: dict[str, str] = {}
    for e in trace.linked_entities:
        if e.qid and e.label:
            out[e.qid] = e.label
    # Scan SPARQL result rows for label-like columns
    for r in trace.results:
        if r.is_empty:
            continue
        for row in r.rows:
            # Pull QID-shaped values together with any *Label sibling column
            for k, v in row.items():
                if not isinstance(v, str):
                    continue
                m = re.fullmatch(r"Q\d+", v)
                if not m:
                    # MaRDI rows often contain full entity URIs
                    m2 = re.search(r"/entity/(Q\d+)$", v)
                    if not m2:
                        continue
                    qid = m2.group(1)
                else:
                    qid = m.group(0)
                # Find a sibling label column (e.g. "item" -> "itemLabel")
                label_col = f"{k}Label"
                if label_col in row and isinstance(row[label_col], str):
                    out.setdefault(qid, row[label_col])
    return out


# ============================================================
# Structural metrics — computed deterministically from the trace
# ============================================================


def structural_metrics(trace: PipelineTrace, latency_s: float, error: Optional[str]) -> dict:
    """Everything we can measure WITHOUT another LLM call (except a cheap API check)."""
    sparql_returned_rows = sum(len(r.rows) for r in trace.results) if trace.results else 0
    answer_words = len(trace.answer.split()) if trace.answer else 0

    # Unique QIDs the model mentioned in its answer
    answer_qids = sorted(set(re.findall(r"\bQ\d+\b", trace.answer or "")))

    # Validity check against MaRDI
    _wbgetentities_batch(answer_qids)
    qids_valid: list[str] = []
    qids_invalid: list[str] = []
    qids_unverified: list[str] = []  # API failed
    for qid in answer_qids:
        info = _qid_info_cache.get(qid, {"exists": None})
        if info["exists"] is True:
            qids_valid.append(qid)
        elif info["exists"] is False:
            qids_invalid.append(qid)
        else:
            qids_unverified.append(qid)

    n_qids = len(answer_qids)
    validity_rate = (len(qids_valid) / n_qids) if n_qids else None

    # Resolve labels for any answer QIDs we could verify: trace first, then cache
    trace_labels = _labels_from_trace(trace)
    qid_labels: dict[str, Optional[str]] = {}
    for qid in answer_qids:
        if qid in trace_labels:
            qid_labels[qid] = trace_labels[qid]
        else:
            qid_labels[qid] = _qid_info_cache.get(qid, {}).get("label")

    return {
        "error": error,
        "latency_seconds": round(latency_s, 2),
        "sparql_attempts": len(trace.queries),
        "sparql_executed_ok": bool(trace.results),
        "sparql_returned_rows": sparql_returned_rows,
        "sparql_errors": list(trace.query_errors),
        "used_fallback": trace.used_fallback,
        "entities_linked": len(trace.linked_entities),
        "answer_length_words": answer_words,
        "answer_qid_mentions": n_qids,
        "answer_qids_unique": answer_qids,
        "answer_qids_valid": qids_valid,
        "answer_qids_invalid": qids_invalid,
        "answer_qids_unverified": qids_unverified,
        "qid_validity_rate": round(validity_rate, 3) if validity_rate is not None else None,
        "qid_labels": qid_labels,
        "answer_non_empty": bool(trace.answer and trace.answer.strip()),
    }


def trace_to_dict(trace: PipelineTrace) -> dict:
    """Serialize the whole PipelineTrace for archival inspection."""
    return {
        "extracted": trace.extracted,
        "linked_entities": [
            {"qid": e.qid, "label": e.label, "description": e.description, "mention": e.mention}
            for e in trace.linked_entities
        ],
        "properties_shown": trace.properties_shown,
        "queries": trace.queries,
        "query_errors": trace.query_errors,
        "results": [
            {"rows": r.rows if not r.is_empty else [], "is_empty": r.is_empty}
            for r in trace.results
        ],
        "used_fallback": trace.used_fallback,
        "fallback_context": trace.fallback_context,
        "answer": trace.answer,
    }


# ============================================================
# LLM-as-judge — Claude compares candidate to baseline
# ============================================================


JUDGE_SYSTEM = """\
You are evaluating two answers to the same question about the MaRDI mathematical
knowledge graph (software packages, publications, authors, mathematical topics).

You will see:
  - The user question
  - BASELINE answer (the reference; the system used to use this)
  - CANDIDATE answer (a different LLM is being evaluated)

Score the CANDIDATE on four 1-4 scales relative to the question:
  - correctness   (1 = wrong / hallucinated, 4 = factually correct)
  - completeness  (1 = misses the main point, 4 = covers all important parts)
  - conciseness   (1 = verbose / rambling, 4 = appropriately brief)
  - faithfulness  (1 = invents data not in retrieved context, 4 = sticks to data)

Then give an OVERALL verdict comparing CANDIDATE to BASELINE:
  - "better"  if candidate is clearly more useful
  - "same"    if roughly equivalent
  - "worse"   if candidate is clearly less useful

Return STRICT JSON, nothing else:
{
  "correctness": <1-4>,
  "completeness": <1-4>,
  "conciseness": <1-4>,
  "faithfulness": <1-4>,
  "verdict": "better" | "same" | "worse",
  "reason": "<one short sentence>"
}
"""


def judge(question: str, baseline_answer: str, candidate_answer: str, judge_llm: LLM) -> dict:
    user = f"""Question:
{question}

BASELINE answer:
{baseline_answer or "(empty)"}

CANDIDATE answer:
{candidate_answer or "(empty)"}
"""
    try:
        return judge_llm.complete_json(JUDGE_SYSTEM, user)
    except Exception as exc:
        return {
            "correctness": None, "completeness": None,
            "conciseness": None, "faithfulness": None,
            "verdict": "error", "reason": f"judge failure: {exc}",
        }


# ----- Separate judge: QID relevance ------------------------------------------

QID_RELEVANCE_SYSTEM = """\
You evaluate whether the MaRDI knowledge-graph item IDs (QIDs, of the form Q12345)
referenced inside an answer are relevant to the user question. Judge on the basis
of the QUESTION and ANSWER only — do NOT consider what the system retrieved.

You will receive:
  - The user question
  - The model's answer (which mentions one or more QIDs)
  - The list of unique QIDs mentioned, with their labels when available

Score qid_relevance on a 1-4 scale:
  - 1 = referenced items are not relevant to the question or the answer's claims
  - 2 = a minority of references are relevant; many are off-topic or unjustified
  - 3 = most references are relevant; a few are tangential
  - 4 = every referenced item is clearly relevant to the question and the answer's claims

Return STRICT JSON, nothing else:
{
  "qid_relevance": <1-4>,
  "reason": "<one short sentence>"
}
"""


def judge_qid_relevance(
    question: str,
    answer: str,
    qid_labels: dict[str, Optional[str]],
    judge_llm: LLM,
) -> dict:
    """Ask the judge to score how relevant the QIDs in the answer are to the question.

    Context intentionally excludes retrieved data — this is a relevance check, not a
    provenance check.
    """
    if not qid_labels:
        return {"qid_relevance": None, "reason": "answer mentions no QIDs"}

    qid_list_lines = []
    for qid, label in qid_labels.items():
        qid_list_lines.append(f"  - {qid}: {label or '(label unknown)'}")
    user = f"""Question:
{question}

Answer:
{answer or "(empty)"}

QIDs referenced in the answer:
{chr(10).join(qid_list_lines)}
"""
    try:
        return judge_llm.complete_json(QID_RELEVANCE_SYSTEM, user)
    except Exception as exc:
        return {"qid_relevance": None, "reason": f"judge failure: {exc}"}


# ============================================================
# Driver
# ============================================================


def safe_model_slug(provider: str, model: str) -> str:
    """Filesystem-safe identifier for a (provider, model) pair."""
    raw = f"{provider}_{model}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw)


def safe_question_slug(idx: int, question: str) -> str:
    short = re.sub(r"[^A-Za-z0-9]+", "_", question)[:40].strip("_")
    return f"Q{idx + 1:02d}_{short}"


def run_one(question: str, llm: LLM, property_catalog: dict[str, str]) -> tuple[PipelineTrace, float, Optional[str]]:
    t0 = time.perf_counter()
    err: Optional[str] = None
    trace = PipelineTrace()
    try:
        trace = answer_question(question, llm, property_catalog)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    return trace, time.perf_counter() - t0, err


def main() -> int:
    # --- Preflight checks ------------------------------------------------------
    if not config.ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY is not set (needed for Claude baseline + judge).", file=sys.stderr)
        return 1
    needs_ollama = any(p == "ollama" for p, _ in CANDIDATES)
    if needs_ollama and not config.OLLAMA_API_KEY:
        print("ERROR: OLLAMA_API_KEY is not set but Ollama candidates are listed.", file=sys.stderr)
        return 1

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = OUTPUT_ROOT / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[init] Output directory: {run_dir}")

    # --- Property catalog (shared by all candidates, fetched once) -------------
    print("[init] Fetching MaRDI property catalog…")
    try:
        property_catalog = fetch_property_catalog()
        print(f"[init] Loaded {len(property_catalog)} properties.")
    except Exception as exc:
        print(f"ERROR: could not load property catalog: {exc}", file=sys.stderr)
        return 1

    # --- Run every (model, question) pair --------------------------------------
    # Storage: per_model[model_label][q_idx] = {trace_dict, metrics}
    per_model: dict[str, dict[int, dict]] = {}
    model_labels: list[str] = []

    for provider, model in CANDIDATES:
        label = f"{provider}/{model}"
        model_labels.append(label)
        slug = safe_model_slug(provider, model)
        model_dir = run_dir / slug
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Model: {label} ===")
        try:
            llm = LLM(provider=provider, model=model)
        except Exception as exc:
            print(f"  [skip] could not init: {exc}")
            per_model[label] = {}
            continue

        per_model[label] = {}
        for i, q in enumerate(QUESTIONS):
            print(f"  [{i + 1}/{len(QUESTIONS)}] {q[:70]}{'…' if len(q) > 70 else ''}")
            trace, latency, err = run_one(q, llm, property_catalog)
            metrics = structural_metrics(trace, latency, err)

            record = {
                "question": q,
                "model": label,
                "metrics": metrics,
                "trace": trace_to_dict(trace),
            }
            per_model[label][i] = record

            q_slug = safe_question_slug(i, q)
            with (model_dir / f"{q_slug}.json").open("w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            status = "OK" if not err else f"ERR ({err.splitlines()[0]})"
            print(
                f"      latency={metrics['latency_seconds']:.1f}s  "
                f"sparql_rows={metrics['sparql_returned_rows']}  "
                f"answer_words={metrics['answer_length_words']}  "
                f"{status}"
            )

    # --- LLM-as-judge: QID relevance (per-model, not pairwise) -----------------
    # We score QID relevance for EVERY model, including the baseline, so that
    # the structural-metrics page can show a fair bar chart.
    print("\n=== QID relevance phase ===")
    qid_judge_llm = LLM(provider=JUDGE_PROVIDER, model=JUDGE_MODEL)
    for label, records in per_model.items():
        if not records:
            continue
        print(f"  Judging QID relevance for {label}…")
        for i, rec in records.items():
            metrics = rec.get("metrics") or {}
            answer = (rec.get("trace") or {}).get("answer", "")
            qid_labels = metrics.get("qid_labels") or {}
            verdict = judge_qid_relevance(rec["question"], answer, qid_labels, qid_judge_llm)
            metrics["qid_relevance_score"] = verdict.get("qid_relevance")
            metrics["qid_relevance_reason"] = verdict.get("reason", "")
            # Re-persist the per-question file with the enriched metrics
            slug_dir = run_dir / safe_model_slug(*label.split("/", 1))
            q_slug = safe_question_slug(i, rec["question"])
            with (slug_dir / f"{q_slug}.json").open("w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
            print(f"    Q{i + 1}: qid_relevance={verdict.get('qid_relevance', '—')}  "
                  f"({(verdict.get('reason') or '')[:70]})")

    # --- LLM-as-judge: compare each candidate answer to the Claude baseline ----
    print("\n=== Judge phase ===")
    baseline_label = model_labels[0]
    candidate_labels = model_labels[1:]

    judge_llm = LLM(provider=JUDGE_PROVIDER, model=JUDGE_MODEL)
    comparisons: dict[str, list[dict]] = {label: [] for label in candidate_labels}

    for cand_label in candidate_labels:
        print(f"  Judging {cand_label} vs {baseline_label}…")
        for i, q in enumerate(QUESTIONS):
            baseline_rec = per_model.get(baseline_label, {}).get(i)
            cand_rec = per_model.get(cand_label, {}).get(i)
            if not baseline_rec or not cand_rec:
                comparisons[cand_label].append({"question": q, "verdict": "missing", "reason": "record not available"})
                continue
            verdict = judge(
                question=q,
                baseline_answer=baseline_rec["trace"]["answer"],
                candidate_answer=cand_rec["trace"]["answer"],
                judge_llm=judge_llm,
            )
            verdict["question"] = q
            comparisons[cand_label].append(verdict)
            print(f"    Q{i + 1}: {verdict.get('verdict', '?')}  ({verdict.get('reason', '')[:80]})")

    comparisons_dir = OUTPUT_ROOT / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    with (comparisons_dir / f"{timestamp}.json").open("w", encoding="utf-8") as f:
        json.dump(comparisons, f, ensure_ascii=False, indent=2)

    # --- Summary ----------------------------------------------------------------
    summary = build_summary(per_model, comparisons, model_labels, baseline_label)
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print_summary(summary)
    print(f"\n[done] Full results: {run_dir}")

    # --- Build PDF report (optional; do not fail eval if it errors) -----------
    try:
        from visualize import build_report
        report_path = build_report(run_dir, OUTPUT_ROOT)
        print(f"[done] PDF report: {report_path}")
    except ImportError:
        print("[viz] Skipping PDF report (matplotlib not installed). "
              "Install with: pip install matplotlib")
    except Exception as exc:
        print(f"[viz] Could not build PDF report: {exc}")
        print("      You can build it later with:  python visualize.py")

    return 0


def build_summary(
    per_model: dict[str, dict[int, dict]],
    comparisons: dict[str, list[dict]],
    model_labels: list[str],
    baseline_label: str,
) -> dict:
    summary: dict = {"baseline": baseline_label, "models": {}}
    for label in model_labels:
        records = list(per_model.get(label, {}).values())
        n = len(records)
        if n == 0:
            summary["models"][label] = {"error": "no records"}
            continue

        sparql_ok = sum(1 for r in records if r["metrics"]["sparql_executed_ok"]
                        and r["metrics"]["sparql_returned_rows"] > 0)
        avg_latency = sum(r["metrics"]["latency_seconds"] for r in records) / n
        avg_answer_len = sum(r["metrics"]["answer_length_words"] for r in records) / n
        non_empty = sum(1 for r in records if r["metrics"]["answer_non_empty"])
        errors = sum(1 for r in records if r["metrics"]["error"])

        # QID validity rate: per question = valid / (valid + invalid); average over questions
        # (questions with no QID mentions are skipped — they have no validity signal)
        validity_per_q = []
        total_qids = 0
        total_invalid = 0
        for r in records:
            m = r["metrics"]
            rate = m.get("qid_validity_rate")
            if rate is not None:
                validity_per_q.append(rate)
            total_qids += m.get("answer_qid_mentions", 0)
            total_invalid += len(m.get("answer_qids_invalid") or [])
        avg_validity = (sum(validity_per_q) / len(validity_per_q)) if validity_per_q else None

        # QID relevance: average over questions where the judge actually scored
        relevance_scores = [
            r["metrics"].get("qid_relevance_score")
            for r in records
            if isinstance(r["metrics"].get("qid_relevance_score"), (int, float))
        ]
        avg_relevance = (sum(relevance_scores) / len(relevance_scores)) if relevance_scores else None

        bucket = {
            "n": n,
            "sparql_success_rate": f"{sparql_ok}/{n}",
            "avg_latency_s": round(avg_latency, 2),
            "avg_answer_words": round(avg_answer_len, 1),
            "non_empty_answers": f"{non_empty}/{n}",
            "pipeline_errors": errors,
            "qid_total_mentions": total_qids,
            "qid_invalid_total": total_invalid,
            "qid_validity_rate": round(avg_validity, 3) if avg_validity is not None else None,
            "qid_relevance_avg": round(avg_relevance, 2) if avg_relevance is not None else None,
            "qid_relevance_n_scored": len(relevance_scores),
        }
        if label != baseline_label:
            verdicts = comparisons.get(label, [])
            counts = {"better": 0, "same": 0, "worse": 0, "error": 0, "missing": 0}
            scores = {"correctness": [], "completeness": [], "conciseness": [], "faithfulness": []}
            for v in verdicts:
                counts[v.get("verdict", "error")] = counts.get(v.get("verdict", "error"), 0) + 1
                for k in scores:
                    val = v.get(k)
                    if isinstance(val, (int, float)):
                        scores[k].append(val)
            bucket["vs_baseline"] = counts
            bucket["avg_scores"] = {
                k: round(sum(v) / len(v), 2) if v else None for k, v in scores.items()
            }
        summary["models"][label] = bucket
    return summary


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print(f"Eval summary — baseline: {summary['baseline']}")
    print("=" * 60)
    for label, bucket in summary["models"].items():
        print(f"\n{label}")
        if "error" in bucket:
            print(f"  ! {bucket['error']}")
            continue
        print(f"  SPARQL success:    {bucket['sparql_success_rate']}")
        print(f"  Non-empty answers: {bucket['non_empty_answers']}")
        print(f"  Pipeline errors:   {bucket['pipeline_errors']}")
        print(f"  Avg latency:       {bucket['avg_latency_s']}s")
        print(f"  Avg answer words:  {bucket['avg_answer_words']}")
        if bucket.get("qid_total_mentions"):
            vr = bucket.get("qid_validity_rate")
            rr = bucket.get("qid_relevance_avg")
            print(f"  QID validity:      "
                  f"{vr * 100:.0f}% valid "
                  f"({bucket['qid_total_mentions'] - bucket['qid_invalid_total']}/"
                  f"{bucket['qid_total_mentions']})" if vr is not None else "  QID validity: n/a")
            print(f"  QID relevance:     "
                  f"{rr:.2f} / 4.0" if rr is not None else "  QID relevance: n/a")
        if "vs_baseline" in bucket:
            vb = bucket["vs_baseline"]
            print(f"  vs Claude:         {vb['better']} better / {vb['same']} same / {vb['worse']} worse"
                  + (f" ({vb['error']} judge errors)" if vb["error"] else ""))
            scores = bucket["avg_scores"]
            print(
                f"  Avg scores (1-4):  "
                f"correctness={scores['correctness']}, "
                f"completeness={scores['completeness']}, "
                f"conciseness={scores['conciseness']}, "
                f"faithfulness={scores['faithfulness']}"
            )
    print("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
