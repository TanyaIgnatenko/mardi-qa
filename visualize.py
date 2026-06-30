"""Build a PDF report from an evaluation run.

Reads:
    eval_results/runs/<timestamp>/summary.json
    eval_results/runs/<timestamp>/<model>/Q*.json
    eval_results/comparisons/<timestamp>.json

Writes:
    eval_results/runs/<timestamp>/report.pdf

Usage:
    # Visualize the most recent run
    python visualize.py

    # Visualize a specific run
    python visualize.py --run eval_results/runs/2026-06-23_15-42-00
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # headless backend; no display required
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================
# Styling
# ============================================================

# A palette that prints well in color AND grayscale.
MODEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

# Light grid, no top/right spines — clean for reports.
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})


# ============================================================
# Data loading
# ============================================================


def find_latest_run(eval_root: Path) -> Optional[Path]:
    runs_dir = eval_root / "runs"
    if not runs_dir.exists():
        return None
    candidates = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)  # timestamp-sortable


def load_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found in {run_dir}")
    with summary_path.open(encoding="utf-8") as f:
        return json.load(f)


def load_comparisons(eval_root: Path, timestamp: str) -> dict:
    path = eval_root / "comparisons" / f"{timestamp}.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_per_question_records(run_dir: Path) -> dict[str, list[dict]]:
    """For each model dir, load all per-question JSON files sorted by Q-index."""
    out: dict[str, list[dict]] = {}
    for model_dir in sorted(run_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        records = []
        for jf in sorted(model_dir.glob("Q*.json")):
            with jf.open(encoding="utf-8") as f:
                records.append(json.load(f))
        if records:
            # Re-derive the "provider/model" label from the first record
            label = records[0].get("model", model_dir.name)
            out[label] = records
    return out


# ============================================================
# Helpers
# ============================================================


def short_label(label: str, max_len: int = 22) -> str:
    """Compact 'provider/model' for axis ticks."""
    if "/" in label:
        label = label.split("/", 1)[1]
    if len(label) > max_len:
        label = label[: max_len - 1] + "…"
    return label


def parse_ratio(text: str) -> float:
    """Turn '4/5' into 0.8. Tolerates None / weird values."""
    if not text or "/" not in text:
        return 0.0
    a, b = text.split("/", 1)
    try:
        a, b = float(a), float(b)
    except ValueError:
        return 0.0
    return a / b if b else 0.0


def model_color(idx: int) -> str:
    return MODEL_COLORS[idx % len(MODEL_COLORS)]


def annotate_bars(ax, bars, fmt: str = "{:.2f}"):
    """Print value above each bar."""
    for b in bars:
        h = b.get_height()
        ax.annotate(
            fmt.format(h),
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )


def add_caption(fig, text: str):
    """Caption strip at the bottom of the figure."""
    fig.text(
        0.5, 0.02, text,
        ha="center", va="bottom",
        fontsize=8, style="italic", color="#555555", wrap=True,
    )


# ============================================================
# Page builders
# ============================================================


def page_title(pdf: PdfPages, summary: dict, run_dir: Path, n_questions: int):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.suptitle("MaRDI QA — Model Evaluation Report", fontsize=18, fontweight="bold", y=0.94)

    timestamp = run_dir.name
    baseline = summary.get("baseline", "(unknown)")
    models = list(summary.get("models", {}).keys())

    body_lines = [
        f"Run timestamp:   {timestamp}",
        f"Baseline model:  {baseline}",
        f"Questions:       {n_questions}",
        f"Models tested:   {len(models)}",
        "",
        "Models in this run:",
    ]
    for m in models:
        marker = "  ← BASELINE" if m == baseline else ""
        body_lines.append(f"   • {m}{marker}")

    body_lines.extend([
        "",
        "Report contents:",
        "   1.  Structural metrics (SPARQL success, latency, answer length)",
        "   1b. Knowledge-graph reference quality (QID validity & relevance)",
        "   2.  LLM-as-judge scores (correctness, completeness, conciseness, faithfulness)",
        "   3.  Verdict vs baseline (better / same / worse)",
        "   4.  Per-question heatmap (correctness scores across models)",
        "   5.  Summary table (all numeric metrics)",
        "   6.  Judge verdicts — detailed reasoning (per model, per question)",
        "   7.  Question-by-question walkthrough (full text + clickable links)",
        "",
        "Notes:",
        "   • Judge is Claude. Self-preference bias is possible when scoring",
        "     Claude's own baseline against candidates.",
        "   • QID validity is checked against the live MaRDI wbgetentities API.",
        "   • QID relevance is a separate judge call given only the question and answer.",
        "   • For ZIB-hosted Ollama models, latency includes network/VPN overhead.",
    ])

    fig.text(
        0.1, 0.83, "\n".join(body_lines),
        fontsize=10, family="monospace", va="top",
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_structural_metrics(pdf: PdfPages, summary: dict):
    models_data = summary.get("models", {})
    labels = list(models_data.keys())
    if not labels:
        return

    sparql_rate = [parse_ratio(models_data[m].get("sparql_success_rate", "0/0")) * 100 for m in labels]
    latency = [models_data[m].get("avg_latency_s") or 0 for m in labels]
    answer_len = [models_data[m].get("avg_answer_words") or 0 for m in labels]

    fig, axes = plt.subplots(3, 1, figsize=(8.27, 11.69))
    fig.suptitle("1. Structural Metrics", fontsize=14, fontweight="bold", y=0.97)

    x = np.arange(len(labels))
    short = [short_label(m) for m in labels]
    colors = [model_color(i) for i in range(len(labels))]

    plots = [
        (axes[0], sparql_rate, "Success rate (%)",
         "SPARQL success rate — % of questions returning non-empty results",
         "{:.0f}%", (0, 105),
         "Higher is better. Measures whether the generated SPARQL executed AND returned at least one row."),
        (axes[1], latency, "Seconds per question",
         "Average latency — end-to-end pipeline time",
         "{:.1f}s", None,
         "Lower is better. Local Ollama models include network/VPN overhead when hosted at ZIB."),
        (axes[2], answer_len, "Words",
         "Average answer length",
         "{:.0f}", None,
         "Context-dependent. Very short answers may miss detail; very long ones may indicate verbosity."),
    ]

    for ax, values, ylabel, title, fmt, ylim, caption in plots:
        bars = ax.bar(x, values, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=20, ha="right", fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        annotate_bars(ax, bars, fmt)
        # Caption uses figure-relative coordinates positioned just below each subplot
        ax.set_xlabel(caption, fontsize=8, style="italic", color="#555555", labelpad=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.85)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_qid_metrics(pdf: PdfPages, summary: dict):
    """Second structural-metrics page: QID validity + QID relevance."""
    models_data = summary.get("models", {})
    labels = list(models_data.keys())
    if not labels:
        return

    validity_pct = []
    relevance = []
    for m in labels:
        d = models_data[m]
        vr = d.get("qid_validity_rate")
        rr = d.get("qid_relevance_avg")
        validity_pct.append(vr * 100 if vr is not None else 0)
        relevance.append(rr if rr is not None else 0)

    has_any_validity = any(models_data[m].get("qid_validity_rate") is not None for m in labels)
    has_any_relevance = any(models_data[m].get("qid_relevance_avg") is not None for m in labels)
    if not has_any_validity and not has_any_relevance:
        return  # no QID activity in this run — skip the page entirely

    fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
    fig.suptitle("1b. Knowledge-Graph Reference Quality (QIDs)",
                 fontsize=14, fontweight="bold", y=0.97)

    x = np.arange(len(labels))
    short = [short_label(m) for m in labels]
    colors = [model_color(i) for i in range(len(labels))]

    # --- QID validity rate ---
    ax = axes[0]
    bars = ax.bar(x, validity_pct, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 120)  # extra headroom so annotations don't touch title
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("Valid QIDs (%)")
    ax.set_title("QID validity rate — % of mentioned items that exist in MaRDI")
    for i, (b, m) in enumerate(zip(bars, labels)):
        d = models_data[m]
        vr = d.get("qid_validity_rate")
        total = d.get("qid_total_mentions", 0)
        if vr is None:
            ax.annotate("n/a (no QIDs)", xy=(b.get_x() + b.get_width() / 2, 2),
                        ha="center", va="bottom", fontsize=7, color="#888")
        else:
            ax.annotate(f"{vr * 100:.0f}% ({total - d.get('qid_invalid_total', 0)}/{total})",
                        xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xlabel(
        "Higher is better. Each QID checked against MaRDI's wbgetentities API. "
        "Invalid QIDs indicate hallucinated identifiers.",
        fontsize=8, style="italic", color="#555555", labelpad=12,
    )

    # --- QID relevance ---
    ax = axes[1]
    bars = ax.bar(x, relevance, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 5.0)  # headroom for two-line annotation
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_ylabel("Relevance (1–4)")
    ax.set_title("QID relevance — judge rating of how relevant referenced items are")
    for b, m in zip(bars, labels):
        d = models_data[m]
        rr = d.get("qid_relevance_avg")
        scored = d.get("qid_relevance_n_scored", 0)
        if rr is None:
            ax.annotate("n/a", xy=(b.get_x() + b.get_width() / 2, 0.1),
                        ha="center", va="bottom", fontsize=7, color="#888")
        else:
            ax.annotate(f"{rr:.2f} (n={scored})",
                        xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xlabel(
        "Higher is better. Judge scored 1–4 on relevance of references; "
        "only the question and answer were shown to the judge.",
        fontsize=8, style="italic", color="#555555", labelpad=12,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.7)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_judge_scores(pdf: PdfPages, summary: dict):
    models_data = summary.get("models", {})
    labels = list(models_data.keys())
    if not labels:
        return

    # Pull avg_scores per candidate; baseline has no scores (not judged against itself).
    metrics = ["correctness", "completeness", "conciseness", "faithfulness"]
    metric_descriptions = {
        "correctness":  "(1) wrong / hallucinated  →  (4) fully correct",
        "completeness": "(1) misses the point  →  (4) covers all parts",
        "conciseness":  "(1) rambling / verbose  →  (4) concise",
        "faithfulness": "(1) invents facts  →  (4) grounded in context",
    }

    data: dict[str, list[Optional[float]]] = {m: [] for m in metrics}
    for model in labels:
        scores = models_data[model].get("avg_scores") or {}
        for m in metrics:
            data[m].append(scores.get(m))

    fig, axes = plt.subplots(2, 2, figsize=(8.27, 11.69))
    fig.suptitle("2. LLM-as-Judge Scores (1–4 scale)", fontsize=14, fontweight="bold", y=0.97)

    x = np.arange(len(labels))
    short = [short_label(m) for m in labels]
    colors = [model_color(i) for i in range(len(labels))]

    for ax, metric in zip(axes.flat, metrics):
        values = [v if v is not None else 0 for v in data[metric]]
        bars = ax.bar(x, values, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 4.3)
        ax.set_ylabel("Score (1–4)")
        ax.set_title(metric.capitalize())
        # Annotate non-baseline bars; baseline has no score
        for b, v, m in zip(bars, data[metric], labels):
            if v is None:
                ax.annotate("baseline", xy=(b.get_x() + b.get_width() / 2, 0.1),
                            ha="center", va="bottom", fontsize=7, color="#888")
            else:
                ax.annotate(f"{v:.2f}", xy=(b.get_x() + b.get_width() / 2, v),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)
        # Caption as xlabel below the tick labels — avoids overlap
        ax.set_xlabel(metric_descriptions[metric],
                      fontsize=7.5, style="italic", color="#555555", labelpad=8)

    add_caption(fig,
                "Baseline (Claude) is not scored against itself. "
                "Scores reflect Claude's judgment of each candidate's answer.")

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.subplots_adjust(hspace=0.7, wspace=0.35)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_vs_baseline(pdf: PdfPages, summary: dict):
    models_data = summary.get("models", {})
    baseline = summary.get("baseline")
    candidates = [m for m in models_data if m != baseline]
    if not candidates:
        return

    better = [models_data[m].get("vs_baseline", {}).get("better", 0) for m in candidates]
    same   = [models_data[m].get("vs_baseline", {}).get("same",   0) for m in candidates]
    worse  = [models_data[m].get("vs_baseline", {}).get("worse",  0) for m in candidates]
    errors = [models_data[m].get("vs_baseline", {}).get("error",  0) for m in candidates]

    fig, ax = plt.subplots(figsize=(8.27, 5.5))
    fig.suptitle(f"3. Verdict vs Baseline ({short_label(baseline) if baseline else 'baseline'})",
                 fontsize=14, fontweight="bold", y=0.97)

    x = np.arange(len(candidates))
    short = [short_label(m) for m in candidates]

    bars_b = ax.bar(x, better, label="better", color="#2ca02c")
    bars_s = ax.bar(x, same,   label="same",   color="#7f7f7f", bottom=better)
    bars_w = ax.bar(x, worse,  label="worse",  color="#d62728",
                    bottom=[b + s for b, s in zip(better, same)])
    if any(errors):
        bottoms = [b + s + w for b, s, w in zip(better, same, worse)]
        ax.bar(x, errors, label="judge error", color="#bbbbbb",
               bottom=bottoms, hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=20, ha="right")
    ax.set_ylabel("Number of questions")
    ax.set_title("Per-question judge verdict, stacked")
    ax.legend(loc="upper right")

    # Annotate counts on each segment
    for bars, vals in [(bars_b, better), (bars_s, same), (bars_w, worse)]:
        for b, v in zip(bars, vals):
            if v > 0:
                ax.annotate(str(v),
                            xy=(b.get_x() + b.get_width() / 2, b.get_y() + b.get_height() / 2),
                            ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    add_caption(fig,
                "Each candidate's answers were compared question-by-question to Claude's. "
                "Stacked counts show how many times the candidate was judged better, same, or worse.")

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_heatmap(pdf: PdfPages, eval_root: Path, run_dir: Path, summary: dict):
    """Heatmap: rows = models (excluding baseline), columns = questions, color = correctness."""
    comparisons = load_comparisons(eval_root, run_dir.name)
    if not comparisons:
        return

    candidate_labels = list(comparisons.keys())
    if not candidate_labels:
        return

    n_questions = max(len(v) for v in comparisons.values())
    matrix = np.full((len(candidate_labels), n_questions), np.nan)
    questions_seen: list[str] = ["" for _ in range(n_questions)]

    for i, label in enumerate(candidate_labels):
        for j, verdict in enumerate(comparisons[label]):
            score = verdict.get("correctness")
            if isinstance(score, (int, float)):
                matrix[i, j] = score
            if not questions_seen[j]:
                questions_seen[j] = verdict.get("question", f"Q{j+1}")

    fig, ax = plt.subplots(figsize=(8.27, 4 + 0.4 * len(candidate_labels)))
    fig.suptitle("4. Per-question Correctness Heatmap", fontsize=14, fontweight="bold", y=0.99)

    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap="RdYlGn", vmin=1, vmax=4, aspect="auto")

    ax.set_xticks(range(n_questions))
    ax.set_xticklabels([f"Q{i+1}" for i in range(n_questions)], fontsize=9)
    ax.set_yticks(range(len(candidate_labels)))
    ax.set_yticklabels([short_label(m, 28) for m in candidate_labels], fontsize=9)
    ax.set_xlabel("Question")

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="#888")
            else:
                # Text color contrasts with cell color
                text_color = "white" if v < 2 or v > 3.5 else "black"
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=9, color=text_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, ticks=[1, 2, 3, 4])
    cbar.set_label("Correctness (1–4)")

    # Question legend at the bottom
    q_lines = ["Questions:"]
    for j, q in enumerate(questions_seen):
        q_short = (q[:90] + "…") if len(q) > 90 else q
        q_lines.append(f"  Q{j+1}: {q_short}")
    fig.text(0.05, 0.02, "\n".join(q_lines), fontsize=7, family="monospace", va="bottom")

    plt.tight_layout(rect=[0, 0.18, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ----- Detailed appendix pages ------------------------------------------------


# A4 portrait page in inches
_PAGE_W = 8.27
_PAGE_H = 11.69

# Color codes for verdicts (used in headings)
_VERDICT_COLORS = {
    "better": "#1b7a1b",
    "same":   "#666666",
    "worse":  "#b02020",
    "error":  "#cc6600",
    "missing": "#888888",
}


def _wrap_text_lines(text: str, max_chars: int) -> list[str]:
    """Very simple paragraph wrapper that respects whitespace; preserves newlines."""
    out: list[str] = []
    for paragraph in (text or "").splitlines() or [""]:
        if not paragraph.strip():
            out.append("")
            continue
        words = paragraph.split(" ")
        line = ""
        for w in words:
            if not line:
                line = w
            elif len(line) + 1 + len(w) <= max_chars:
                line += " " + w
            else:
                out.append(line)
                line = w
        if line:
            out.append(line)
    return out


def _new_appendix_page(title: str):
    """Create a new figure formatted as a portrait appendix page."""
    fig = plt.figure(figsize=(_PAGE_W, _PAGE_H))
    fig.text(0.5, 0.965, title, ha="center", va="top",
             fontsize=14, fontweight="bold")
    fig.text(0.08, 0.935, "_" * 95, ha="left", va="top",
             fontsize=6, color="#cccccc")
    return fig


def _flush_page(pdf: PdfPages, fig):
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_judge_reasons(pdf: PdfPages, comparisons: dict, baseline_label: str):
    """Per-model breakdown of judge reasoning — full text, no truncation."""
    if not comparisons:
        return

    # We'll lay out by iterating through entries and starting a new page when
    # we run out of vertical space. Y-coordinates in figure-fraction.
    y_top = 0.92
    y_bottom = 0.06
    line_h = 0.014  # vertical fraction per line of body text
    max_chars_body = 110  # rough chars per line at 9pt monospace

    fig = _new_appendix_page("6. Judge Verdicts — Detailed Reasoning")
    y = y_top

    def new_page_if_needed(needed_lines: int):
        nonlocal fig, y
        needed_space = needed_lines * line_h + 0.005
        if y - needed_space < y_bottom:
            _flush_page(pdf, fig)
            fig = _new_appendix_page("6. Judge Verdicts — Detailed Reasoning (cont.)")
            y = y_top

    for cand_label, verdicts in comparisons.items():
        # Section header per candidate model
        new_page_if_needed(3)
        fig.text(0.08, y, cand_label, fontsize=11, fontweight="bold", color="#222266")
        y -= line_h * 1.6
        fig.text(0.08, y, "—" * 100, fontsize=6, color="#cccccc")
        y -= line_h * 0.7

        for i, v in enumerate(verdicts, start=1):
            q = v.get("question", f"Q{i}")
            verdict = v.get("verdict", "?")
            reason = v.get("reason", "")
            scores = (
                f"correctness={v.get('correctness', '—')}  "
                f"completeness={v.get('completeness', '—')}  "
                f"conciseness={v.get('conciseness', '—')}  "
                f"faithfulness={v.get('faithfulness', '—')}"
            )

            q_lines = _wrap_text_lines(f"Q{i}: {q}", max_chars_body)
            r_lines = _wrap_text_lines(f"Reason: {reason}", max_chars_body)
            # Reserve room for the question + verdict/scores line; the reason
            # itself may span pages if exceptionally long.
            new_page_if_needed(len(q_lines) + 2)

            # Question line(s)
            for ln in q_lines:
                fig.text(0.10, y, ln, fontsize=9, fontweight="bold", color="#111")
                y -= line_h

            # Verdict + scores line
            color = _VERDICT_COLORS.get(verdict, "#000")
            fig.text(0.12, y, f"Verdict: {verdict.upper()}", fontsize=9,
                     color=color, fontweight="bold")
            fig.text(0.32, y, scores, fontsize=8, color="#555", family="monospace")
            y -= line_h * 1.05

            # Reason lines — overflow to a new page if needed
            for ln in r_lines:
                if y - line_h < y_bottom:
                    _flush_page(pdf, fig)
                    fig = _new_appendix_page("6. Judge Verdicts — Detailed Reasoning (cont.)")
                    y = y_top
                    fig.text(0.10, y, f"  ↳ {cand_label} — Q{i} (continued)",
                             fontsize=8.5, style="italic", color="#888")
                    y -= line_h * 1.2
                fig.text(0.12, y, ln, fontsize=8.5, color="#333")
                y -= line_h
            y -= line_h * 0.5  # padding between verdicts

        y -= line_h * 0.7  # padding between models

    _flush_page(pdf, fig)


def page_question_walkthrough(
    pdf: PdfPages,
    per_model_records: dict[str, list[dict]],
    questions: list[str],
    baseline_label: str,
    comparisons: dict,
    preferred_second: str = "ollama/qwen3-coder-next:latest",
):
    """For each question, print the question and then every model's full answer.

    Ordering rule: baseline first, then `preferred_second` if it exists, then
    the rest in their original order. Each non-baseline model header includes
    the judge's verdict (BETTER / SAME / WORSE) coloured accordingly.
    """
    if not per_model_records or not questions:
        return

    # Determine model ordering
    available = list(per_model_records.keys())
    ordered: list[str] = []
    if baseline_label in available:
        ordered.append(baseline_label)
    if preferred_second in available and preferred_second not in ordered:
        ordered.append(preferred_second)
    for m in available:
        if m not in ordered:
            ordered.append(m)

    # Build {question_idx: {model_label: record}}
    by_q: dict[int, dict[str, dict]] = {}
    for label, recs in per_model_records.items():
        for rec in recs:
            try:
                qi = questions.index(rec.get("question", ""))
            except ValueError:
                continue
            by_q.setdefault(qi, {})[label] = rec

    # Build a quick lookup: {(model_label, question_idx): verdict_str}
    verdict_lookup: dict[tuple[str, int], str] = {}
    for model_label, verdicts in (comparisons or {}).items():
        for v in verdicts:
            try:
                qi = questions.index(v.get("question", ""))
            except ValueError:
                continue
            verdict_lookup[(model_label, qi)] = v.get("verdict", "")

    # Layout parameters
    y_top = 0.92
    y_bottom = 0.06
    line_h = 0.014
    max_chars_body = 110

    fig = _new_appendix_page("7. Question-by-Question Walkthrough")
    y = y_top

    def new_page_if_needed(needed_lines: int):
        nonlocal fig, y
        needed_space = needed_lines * line_h + 0.005
        if y - needed_space < y_bottom:
            _flush_page(pdf, fig)
            fig = _new_appendix_page("7. Question-by-Question Walkthrough (cont.)")
            y = y_top

    for qi, q in enumerate(questions):
        recs = by_q.get(qi, {})
        # Question header
        q_lines = _wrap_text_lines(f"Q{qi + 1}: {q}", max_chars_body)
        new_page_if_needed(len(q_lines) + 3)

        for ln in q_lines:
            fig.text(0.08, y, ln, fontsize=10.5, fontweight="bold", color="#111")
            y -= line_h * 1.15
        fig.text(0.08, y, "—" * 100, fontsize=6, color="#bbb")
        y -= line_h * 0.7

        # Each model's answer in determined order
        for label in ordered:
            rec = recs.get(label)
            if not rec:
                continue
            answer = (rec.get("trace") or {}).get("answer", "") or "(no answer)"
            metrics = rec.get("metrics") or {}
            meta_line = (
                f"latency={metrics.get('latency_seconds', '?')}s  "
                f"sparql_rows={metrics.get('sparql_returned_rows', '?')}  "
                f"words={metrics.get('answer_length_words', '?')}"
            )

            ans_lines = _wrap_text_lines(answer, max_chars_body)
            needed = len(ans_lines) + 3
            new_page_if_needed(needed)

            # Model header — with verdict tag for non-baseline.
            # Verdict tag is right-aligned to avoid overlapping the model name
            # (which varies in length).
            is_base = (label == baseline_label)
            header_text = f"▸ {label}"
            header_color = "#1f4e8c" if is_base else "#444"
            new_page_if_needed(3)
            fig.text(0.10, y, header_text, fontsize=9.5, fontweight="bold", color=header_color)

            if is_base:
                fig.text(0.92, y, "(BASELINE)",
                         fontsize=9.5, fontweight="bold", color="#1f4e8c",
                         ha="right")
            else:
                verdict = verdict_lookup.get((label, qi), "")
                if verdict:
                    color = _VERDICT_COLORS.get(verdict, "#000")
                    fig.text(0.92, y, f"({verdict.upper()})",
                             fontsize=9.5, fontweight="bold", color=color,
                             ha="right")

            y -= line_h * 1.15
            fig.text(0.12, y, meta_line, fontsize=7.5, color="#777", family="monospace")
            y -= line_h * 1.1

            # Answer body — full text, no truncation.
            # Print one line at a time and start a new page mid-answer if we
            # run out of vertical space, so even very long answers fit fully.
            for ln in ans_lines:
                if y - line_h < y_bottom:
                    _flush_page(pdf, fig)
                    fig = _new_appendix_page("7. Question-by-Question Walkthrough (cont.)")
                    y = y_top
                    # Continuation marker for the same answer
                    fig.text(0.10, y, f"  ↳ {label} (continued)",
                             fontsize=8.5, style="italic", color="#888")
                    y -= line_h * 1.2
                fig.text(0.12, y, ln, fontsize=8.5, color="#222")
                y -= line_h
            y -= line_h * 0.35

            # --- Referenced MaRDI items (clickable links) -----------------------
            qids_unique = metrics.get("answer_qids_unique") or []
            qids_invalid = set(metrics.get("answer_qids_invalid") or [])
            qid_labels_map = metrics.get("qid_labels") or {}
            if qids_unique:
                header_text = f"Referenced MaRDI items ({len(qids_unique)}):"
                new_page_if_needed(len(qids_unique) + 2)
                fig.text(0.12, y, header_text,
                         fontsize=8, style="italic", color="#555")
                y -= line_h
                for qid in qids_unique:
                    if y - line_h < y_bottom:
                        _flush_page(pdf, fig)
                        fig = _new_appendix_page("7. Question-by-Question Walkthrough (cont.)")
                        y = y_top
                    label_text = qid_labels_map.get(qid) or "(label unknown)"
                    url = f"https://portal.mardi4nfdi.de/wiki/Item:{qid}"
                    is_invalid = qid in qids_invalid
                    if is_invalid:
                        # Strikethrough-ish styling for invalid QIDs
                        link_text = f"  {qid} — {label_text}"
                        fig.text(0.14, y, link_text,
                                 fontsize=8, color="#999", url=url)
                        fig.text(0.62, y, "⚠ not found in MaRDI",
                                 fontsize=7.5, color="#b02020", style="italic")
                    else:
                        # Clickable underlined link (blue, classic hyperlink look)
                        link_text = f"  {qid} — {label_text}"
                        fig.text(0.14, y, link_text,
                                 fontsize=8, color="#1f4e8c", url=url)
                    y -= line_h
            y -= line_h * 0.7

        y -= line_h * 0.7

    _flush_page(pdf, fig)


def page_summary_table(pdf: PdfPages, summary: dict):
    models_data = summary.get("models", {})
    baseline = summary.get("baseline")
    if not models_data:
        return

    headers = [
        "Model", "SPARQL ok", "Latency", "Words",
        "Correct.", "Complete.", "Concise.", "Faith.",
        "QID valid", "QID relev.",
        "vs Claude\n(B/S/W)",
    ]
    rows = []
    for label, data in models_data.items():
        is_base = (label == baseline)
        scores = data.get("avg_scores") or {}
        vb = data.get("vs_baseline") or {}
        bsw = "—" if is_base else f"{vb.get('better', 0)}/{vb.get('same', 0)}/{vb.get('worse', 0)}"
        model_name = short_label(label, 22) + (" (baseline)" if is_base else "")
        vr = data.get("qid_validity_rate")
        rr = data.get("qid_relevance_avg")
        rows.append([
            model_name,
            data.get("sparql_success_rate", "—"),
            f"{data.get('avg_latency_s', 0):.1f}s",
            f"{data.get('avg_answer_words', 0):.0f}",
            "—" if is_base else f"{scores.get('correctness', 0) or 0:.2f}",
            "—" if is_base else f"{scores.get('completeness', 0) or 0:.2f}",
            "—" if is_base else f"{scores.get('conciseness', 0) or 0:.2f}",
            "—" if is_base else f"{scores.get('faithfulness', 0) or 0:.2f}",
            "n/a" if vr is None else f"{vr * 100:.0f}%",
            "n/a" if rr is None else f"{rr:.2f}",
            bsw,
        ])

    fig, ax = plt.subplots(figsize=(11.69, 6.5))  # A4 landscape
    fig.suptitle("6. Summary Table", fontsize=14, fontweight="bold", y=0.95)

    ax.axis("off")
    col_widths = [0.20, 0.07, 0.07, 0.06, 0.07, 0.07, 0.07, 0.07, 0.08, 0.08, 0.09]
    table = ax.table(
        cellText=rows, colLabels=headers,
        cellLoc="center", loc="upper center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    # Header styling
    for j in range(len(headers)):
        cell = table[(0, j)]
        cell.set_text_props(fontweight="bold", color="white")
        cell.set_facecolor("#4a4a4a")
        cell.set_height(0.10)

    # Alternate row shading, left-align model column
    for i in range(1, len(rows) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f5f5f5")
            if j == 0:
                table[(i, j)].set_text_props(ha="left")
                table[(i, j)]._loc = "left"

    add_caption(fig,
                "Columns vs Claude show better/same/worse counts. "
                "Em-dash (—) indicates a metric that does not apply (baseline is not judged against itself).")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================


def build_report(run_dir: Path, eval_root: Path) -> Path:
    summary = load_summary(run_dir)
    records = load_per_question_records(run_dir)
    comparisons = load_comparisons(eval_root, run_dir.name)
    baseline = summary.get("baseline", "")
    # Sanity: count questions
    n_questions = max((len(v) for v in records.values()), default=0)

    # Canonical question order — take it from baseline records if present, else any
    canonical_records = records.get(baseline) or next(iter(records.values()), [])
    questions = [r.get("question", "") for r in canonical_records]

    out_path = run_dir / "report.pdf"
    print(f"[viz] Building {out_path}")
    with PdfPages(out_path) as pdf:
        page_title(pdf, summary, run_dir, n_questions)
        page_structural_metrics(pdf, summary)
        page_qid_metrics(pdf, summary)
        page_judge_scores(pdf, summary)
        page_vs_baseline(pdf, summary)
        page_heatmap(pdf, eval_root, run_dir, summary)
        page_summary_table(pdf, summary)
        page_judge_reasons(pdf, comparisons, baseline)
        page_question_walkthrough(pdf, records, questions, baseline, comparisons)

        # Embed metadata
        info = pdf.infodict()
        info["Title"] = "MaRDI QA — Model Evaluation Report"
        info["Subject"] = f"Evaluation run {run_dir.name}"
        info["Creator"] = "evaluate.py + visualize.py"

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run",
        type=Path,
        default=None,
        help="Path to a run directory (default: most recent under eval_results/runs/).",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("eval_results"),
        help="Path to the eval_results root (default: ./eval_results).",
    )
    args = parser.parse_args()

    eval_root = args.eval_root
    if args.run:
        run_dir = args.run
        if not run_dir.exists():
            print(f"ERROR: run directory not found: {run_dir}", file=sys.stderr)
            return 1
    else:
        run_dir = find_latest_run(eval_root)
        if not run_dir:
            print(f"ERROR: no runs found under {eval_root}/runs/", file=sys.stderr)
            return 1
        print(f"[viz] Using most recent run: {run_dir}")

    try:
        out_path = build_report(run_dir, eval_root)
    except Exception as exc:
        print(f"ERROR: failed to build report: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    print(f"[done] Report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
