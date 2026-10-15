"""Structural-output gate: can a model emit what the pipeline needs at all?

Run before comparing quality. The pipeline requires parseable JSON from the
extraction step and syntactically valid SPARQL from the generation step; a model
that fails either is unusable here however well it writes prose. One model in an
earlier comparison returned empty answers on all five questions, and finding that
out cost a full quality run — this costs two calls per question.

Only the two structured stages run, not the pipeline, so a failure is attributed
to the model rather than to routing, linking or the endpoint.

    python eval/gate_models.py --out eval/reports/gate.json
    python eval/gate_models.py --models qwen3.5:9b,gemma4:31b
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

from mardi_qa import config, prompts
from mardi_qa.entity_linking import CORE_PROPERTIES
from mardi_qa.llm import LLM
from mardi_qa.sparql_client import SparqlError, run_query, sanitize_query

CANDIDATES = [
    "qwen3-coder-next:latest",   # 79.7B, code — the incumbent
    "deepseek-coder-v2:latest",  # 15.7B, code — the same job five times smaller
    "gemma4:31b",                # 31.3B, general
    "qwen3.5:35b",               # 36.0B, returned nothing in an earlier run
    "deepseek-r1:14b",           # 14.8B, reasoning
    "qwen3.5:9b",                # 9.7B, the floor
]

# Three questions, deliberately of different shapes: a count, a list and a
# two-entity intersection. One is not enough — a model can parse one JSON by luck.
PROBES = [
    ("count", "How many publications in MaRDI used the software BayesDA?",
     [("Q22960", "BayesDA", "R package for Bayesian Data Analysis")]),
    ("list", "Which software in MaRDI implements LU decomposition?",
     [("Q6825481", "LU decomposition", "matrix factorization algorithm")]),
    ("join", "How many publications in MaRDI used both glmnet and glasso?",
     [("Q20169", "glmnet", "Lasso and Elastic-Net Regularized GLMs"),
      ("Q19462", "glasso", "Graphical Lasso")]),
]


def check_extraction(llm: LLM, question: str) -> dict:
    """Stage 1: must return JSON carrying the keys the pipeline reads."""
    started = time.time()
    try:
        raw = llm.complete(prompts.ENTITY_EXTRACTION_SYSTEM, question, max_tokens=800)
    except Exception as exc:
        return {"ok": False, "why": f"{type(exc).__name__}: {str(exc)[:80]}",
                "latency_s": round(time.time() - started, 1)}

    latency = round(time.time() - started, 1)
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return {"ok": False, "why": "no JSON object in reply", "latency_s": latency,
                "sample": raw[:120]}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return {"ok": False, "why": f"unparseable JSON: {exc}", "latency_s": latency,
                "sample": raw[:120]}

    missing = [k for k in ("entities", "keywords") if k not in data]
    if missing:
        return {"ok": False, "why": f"missing keys: {missing}", "latency_s": latency}
    if not isinstance(data.get("entities"), list):
        return {"ok": False, "why": "'entities' is not a list", "latency_s": latency}
    return {"ok": True, "latency_s": latency, "entities": len(data["entities"]),
            "target_type": data.get("target_type")}


def check_generation(llm: LLM, question: str, entities: list[tuple]) -> dict:
    """Stage 2: must return SPARQL the endpoint will parse.

    Syntax only. A query that runs and returns the wrong rows passes this gate —
    that is a quality question, measured separately. What fails here is a reply
    the endpoint rejects outright, which no amount of prompting downstream fixes.
    """
    entity_block = "\n".join(f'- {qid}: "{label}" — {desc}' for qid, label, desc in entities)
    schema_block = "\n".join(f"  wdt:{pid}  {label}" for pid, label in CORE_PROPERTIES.items())
    user = (f"Question: {question}\n\nLinked MaRDI entities:\n{entity_block}\n\n"
            f"Schema of these entities — the ONLY properties you may use:\n"
            f"{entities[0][1]} → X (this entity is the SUBJECT):\n{schema_block}\n"
            f"X → {entities[0][1]} (this entity is the OBJECT):\n{schema_block}\n")

    started = time.time()
    try:
        raw = llm.complete(prompts.SPARQL_GENERATION_SYSTEM, user, max_tokens=1500)
    except Exception as exc:
        return {"ok": False, "why": f"{type(exc).__name__}: {str(exc)[:80]}",
                "latency_s": round(time.time() - started, 1)}
    latency = round(time.time() - started, 1)

    query = sanitize_query(raw)
    if "SELECT" not in query.upper() and "ASK" not in query.upper():
        return {"ok": False, "why": "no SELECT/ASK in reply", "latency_s": latency,
                "sample": raw[:120]}

    try:
        result = run_query(query)
        return {"ok": True, "latency_s": latency, "rows": len(result.rows),
                "query": query[query.find("SELECT"):][:160]}
    except SparqlError as exc:
        text = str(exc)
        # A rejected query is a syntax failure; a timeout or server error is the
        # endpoint's problem and says nothing about the model.
        syntax = "400" in text or "MalformedQuery" in text or "parse" in text.lower()
        return {"ok": not syntax, "why": ("endpoint rejected the query: " if syntax
                                          else "endpoint problem, not the model: ") + text[:90],
                "latency_s": latency, "query": query[query.find("SELECT"):][:160]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(CANDIDATES))
    ap.add_argument("--out", default="eval/reports/gate.json")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    print(f"Structural gate — {len(models)} models x {len(PROBES)} questions "
          f"x 2 stages = {len(models) * len(PROBES) * 2} calls\n")

    results = []
    for model in models:
        print(f"=== {model}", flush=True)
        try:
            llm = LLM(provider="ollama", model=model)
        except Exception as exc:
            print(f"    could not initialise: {type(exc).__name__}: {exc}")
            results.append({"model": model, "error": str(exc)[:120]})
            continue

        # Warm the model before measuring anything. A cold model spends minutes
        # loading into memory, and the first probes then time out and are scored
        # as failures — qwen3-coder-next looked like 2/3 and 78 seconds a call
        # until the trace showed two 180-second load timeouts followed by replies
        # in 1.8 seconds. Loading time is the server's property, not the model's,
        # and must not land in either column.
        # Retried, because the warm-up itself can outlast the per-call timeout:
        # loading the 79.7B model took longer than 180 seconds, so one attempt
        # returns a timeout while the load is still in progress. Three attempts
        # give it up to nine minutes to become resident.
        print("    warming up…", end=" ", flush=True)
        warm_started = time.time()
        warm_error = None
        for attempt in range(1, 4):
            try:
                llm.complete("Reply with the single word OK.", "Say OK", max_tokens=8)
                print(f"ready in {time.time() - warm_started:.0f}s "
                      f"({attempt} attempt{'s' if attempt > 1 else ''})", flush=True)
                warm_error = None
                break
            except Exception as exc:
                warm_error = f"{type(exc).__name__}: {str(exc)[:70]}"
                print(f"[{attempt}]", end=" ", flush=True)
        if warm_error:
            print(f"never became ready after {time.time() - warm_started:.0f}s — "
                  f"{warm_error}", flush=True)

        row = {"model": model, "warm_up_s": round(time.time() - warm_started, 1),
               "warm_up_error": warm_error, "probes": []}
        for shape, question, entities in PROBES:
            ext = check_extraction(llm, question)
            gen = check_generation(llm, question, entities)
            row["probes"].append({"shape": shape, "extraction": ext, "generation": gen})
            print(f"    {shape:<6} json={'ok ' if ext['ok'] else 'FAIL'} "
                  f"({ext['latency_s']}s)  sparql={'ok ' if gen['ok'] else 'FAIL'} "
                  f"({gen['latency_s']}s)"
                  + (f"  rows={gen['rows']}" if gen.get("rows") is not None else ""),
                  flush=True)
            for stage, res in (("json", ext), ("sparql", gen)):
                if not res["ok"]:
                    print(f"           {stage}: {res.get('why', '')}", flush=True)
        results.append(row)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"), "results": results},
            indent=2, ensure_ascii=False), encoding="utf-8")

    _summary(results)
    print(f"\nReport: {args.out}")
    return 0


def _summary(results: list[dict]) -> None:
    print(f"\n{'=' * 72}")
    print(f"{'model':<26} {'json':>6} {'sparql':>7} {'median latency':>15}  verdict")
    for row in results:
        if "probes" not in row:
            print(f"{row['model']:<26} {'—':>6} {'—':>7} {'—':>15}  could not run")
            continue
        n = len(row["probes"])
        j = sum(p["extraction"]["ok"] for p in row["probes"])
        s = sum(p["generation"]["ok"] for p in row["probes"])
        lat = sorted(p["extraction"]["latency_s"] + p["generation"]["latency_s"]
                     for p in row["probes"])[n // 2]
        verdict = "PASSES" if j == n and s == n else "FAILS THE GATE"
        print(f"{row['model']:<26} {j}/{n:<4} {s}/{n:<5} {lat:>14.1f}s  {verdict}")
    print("\n  A model failing either column is excluded from the quality comparison:\n"
          "  the pipeline cannot use it, whatever it scores elsewhere.")


if __name__ == "__main__":
    raise SystemExit(main())
