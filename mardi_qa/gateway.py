"""Routing gateway — the single entry point for answering a user question.

Decides per question which route to take (see router.py):
  sparql — the full KG pipeline in pipeline.answer_question
  rag    — BM25 retrieval over the package corpus + grounded synthesis
  direct — one cheap-LLM call with honesty guardrails, no retrieval

All routes return the same PipelineTrace so the UI and the regression runner
need no route-specific handling. RAG retrieval results are encoded as a
SparqlResult table (qid/label/description/score) for the same reason.
"""

from __future__ import annotations

import time

from . import config, prompts
from .entity_linking import entity_schema
from .llm import LLM
from .pipeline import PipelineTrace, answer_question
from .router import RouteDecision, decide_route
from .sparql_client import SparqlResult
from .telemetry import log_request


def make_router_llm(main_llm: LLM) -> LLM | None:
    """A cheap classifier/direct-answer model on the same provider as the main LLM."""
    if not config.ROUTER_ENABLED:
        return None
    try:
        if main_llm.provider == "anthropic":
            return LLM(provider="anthropic", model=config.ROUTER_ANTHROPIC_MODEL)
        return LLM(
            provider="ollama",
            model=config.ROUTER_OLLAMA_MODEL or main_llm.model,
        )
    except Exception:
        return None  # router LLM unavailable -> heuristic routing still works


def route_and_answer(
    question: str,
    llm: LLM,
    property_catalog: dict[str, str],
    progress=lambda msg: None,
    router_llm: LLM | None = None,
) -> PipelineTrace:
    if router_llm is None:
        router_llm = make_router_llm(llm)

    if config.ROUTER_ENABLED:
        progress("Routing the question…")
        decision = decide_route(question, router_llm)
    else:
        decision = RouteDecision("sparql", "router disabled", "")
    progress(f"Route: {decision.route} — {decision.reason}")

    started = time.time()
    if decision.route == "direct":
        trace = _answer_direct(question, router_llm or llm)
    elif decision.route == "rag":
        trace = _answer_rag(question, llm, progress)
    else:
        trace = answer_question(question, llm, property_catalog, progress)

    trace.route = decision.route
    trace.route_reason = decision.reason
    trace.route_source = decision.source

    # Single logging point: every route returns through here, so coverage is
    # complete without repeating the call in each branch.
    log_request(trace, latency_s=time.time() - started,
                schema_size=_schema_size(trace))
    return trace


def _schema_size(trace: PipelineTrace) -> int | None:
    """Total properties across the schemas shown for this question."""
    if not trace.linked_entities:
        return None
    total = 0
    for e in trace.linked_entities[:3]:
        schema = entity_schema(e.qid)      # cached; no extra endpoint traffic
        total += len(schema["out"]) + len(schema["in"])
    return total


# ----- direct route -------------------------------------------------------------


def _answer_direct(question: str, cheap_llm: LLM) -> PipelineTrace:
    trace = PipelineTrace()
    trace.answer = cheap_llm.complete(
        prompts.DIRECT_ANSWER_SYSTEM, question, max_tokens=700
    )
    return trace


# ----- rag route ----------------------------------------------------------------


def _answer_rag(question: str, llm: LLM, progress) -> PipelineTrace:
    from .rag import get_index  # deferred: corpus load is heavy

    trace = PipelineTrace()
    try:
        index = get_index(progress)
    except Exception as exc:
        # Corpus unavailable (endpoint down, no cache) — degrade to the SPARQL
        # pipeline's honest failure text rather than crashing the request.
        trace.answer = (
            "The package-corpus index is unavailable "
            f"({type(exc).__name__}: {exc}), so this topic search cannot run. "
            "Please try again later."
        )
        trace.query_errors.append(str(exc))
        return trace

    progress("Retrieving matching packages (BM25)…")
    docs = index.search(question, k=config.RAG_TOP_K)
    total = index.match_count(question)

    trace.queries.append(f"# BM25 retrieval over the package corpus\n# query: {question}")
    trace.results.append(SparqlResult(
        columns=["item", "itemLabel", "description", "score"],
        rows=[
            {"item": d.qid, "itemLabel": d.label,
             "description": d.description, "score": str(d.score)}
            for d in docs
        ],
    ))

    if not docs:
        trace.answer = (
            "No packages in the MaRDI corpus matched this topic. "
            "The corpus covers R/CRAN package names and descriptions; "
            "try different wording."
        )
        return trace

    progress("Composing the answer…")
    doc_lines = "\n".join(
        f"- {d.qid} \"{d.label}\": {d.description or '(no description)'}"
        for d in docs
    )
    trace.answer = llm.complete(
        prompts.RAG_ANSWER_SYSTEM.replace("{total}", str(total)),
        f"Question: {question}\n\nTop retrieved packages:\n{doc_lines}",
        max_tokens=1000,
    )
    return trace
