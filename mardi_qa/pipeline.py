"""End-to-end QA pipeline over the MaRDI knowledge graph.

Stages:
  1. Entity extraction (LLM)        -> mentions + topic keywords
  2. Entity linking (Wikibase API)  -> QIDs with labels/descriptions
  3. SPARQL generation (LLM)        -> grounded in real PIDs/QIDs
  4. Execution (+ one repair retry)
  5. Fallback: entity descriptions as raw context if queries fail
  6. Answer generation (LLM)        -> grounded, with QID references
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import prompts
from .entity_linking import (
    LinkedEntity,
    describe_entity,
    pick_best_candidate,
    relevant_properties,
    search_entities,
)
from .llm import LLM
from .sparql_client import SparqlError, SparqlResult, run_query, sanitize_query


@dataclass
class PipelineTrace:
    """Everything the UI needs to show the user what happened."""
    extracted: dict = field(default_factory=dict)
    linked_entities: list[LinkedEntity] = field(default_factory=list)
    properties_shown: dict[str, str] = field(default_factory=dict)
    queries: list[str] = field(default_factory=list)
    query_errors: list[str] = field(default_factory=list)
    results: list[SparqlResult] = field(default_factory=list)
    used_fallback: bool = False
    fallback_context: str = ""
    answer: str = ""


def _format_entities(entities: list[LinkedEntity]) -> str:
    if not entities:
        return "(none linked)"
    return "\n".join(
        f"- {e.qid}: \"{e.label}\" — {e.description or 'no description'}"
        f" (matched mention: {e.mention})"
        for e in entities
    )


def _format_properties(props: dict[str, str]) -> str:
    return "\n".join(f"- wdt:{pid}: {label}" for pid, label in props.items())


def answer_question(
    question: str,
    llm: LLM,
    property_catalog: dict[str, str],
    progress=lambda msg: None,
) -> PipelineTrace:
    trace = PipelineTrace()

    # 1. Extract mentions + keywords -------------------------------------------
    progress("Extracting entities from the question…")
    try:
        trace.extracted = llm.complete_json(prompts.ENTITY_EXTRACTION_SYSTEM, question)
    except Exception:
        trace.extracted = {"entities": [], "keywords": []}

    # 2. Link to MaRDI QIDs ------------------------------------------------------
    progress("Linking entities to MaRDI items…")
    for mention in trace.extracted.get("entities", [])[:5]:
        try:
            best = pick_best_candidate(search_entities(mention))
        except Exception:
            best = None
        if best:
            trace.linked_entities.append(best)

    # 3. Generate SPARQL ---------------------------------------------------------
    progress("Generating SPARQL query…")
    trace.properties_shown = relevant_properties(property_catalog, question)

    gen_user = f"""Question: {question}

Linked MaRDI entities:
{_format_entities(trace.linked_entities)}

Topic keywords (for mwapi full-text search, if needed): {trace.extracted.get("keywords", [])}

Property catalog (PID: label) — use ONLY these:
{_format_properties(trace.properties_shown)}
"""
    query = sanitize_query(llm.complete(prompts.SPARQL_GENERATION_SYSTEM, gen_user))
    trace.queries.append(query)

    # 4. Execute, retry once on failure/empty -----------------------------------
    progress("Running SPARQL against the MaRDI endpoint…")
    result = _try_run(query, trace)

    if result is None or result.is_empty:
        progress("No results — inspecting entity claims and retrying…")
        entity_dump = "\n\n".join(
            f"Claims of {e.qid} ({e.label}):\n{_safe_describe(e.qid)}"
            for e in trace.linked_entities[:3]
        )
        repair_user = f"""{gen_user}

Previous query:
{query}

Outcome: {trace.query_errors[-1] if trace.query_errors else "query ran but returned 0 rows"}

Actual claims of the linked entities (use these property IDs!):
{entity_dump or "(no entities linked — rely on mwapi search)"}
"""
        query2 = sanitize_query(llm.complete(prompts.SPARQL_REPAIR_SYSTEM, repair_user))
        trace.queries.append(query2)
        result = _try_run(query2, trace)

        # 5. Fallback: answer straight from entity descriptions ------------------
        if (result is None or result.is_empty) and trace.linked_entities:
            trace.used_fallback = True
            trace.fallback_context = entity_dump

    # 6. Final answer -------------------------------------------------------------
    progress("Composing the answer…")
    retrieved_parts = []
    for q, r in zip(trace.queries, trace.results):
        retrieved_parts.append(f"SPARQL query:\n{q}\n\nResults:\n{r.to_prompt_text()}")
    if trace.used_fallback:
        retrieved_parts.append(
            "Fallback — full claims of linked entities:\n" + trace.fallback_context
        )

    retrieved_text = "\n\n---\n\n".join(retrieved_parts) if retrieved_parts else "(nothing retrieved)"
    answer_user = f"""User question: {question}

Linked entities:
{_format_entities(trace.linked_entities)}

Retrieved knowledge-graph data:
{retrieved_text}
"""
    trace.answer = llm.complete(prompts.ANSWER_SYSTEM, answer_user, max_tokens=1200)
    return trace


def _try_run(query: str, trace: PipelineTrace) -> SparqlResult | None:
    try:
        result = run_query(query)
        trace.results.append(result)
        return result
    except SparqlError as exc:
        trace.query_errors.append(str(exc))
        return None


def _safe_describe(qid: str) -> str:
    try:
        return describe_entity(qid)
    except Exception as exc:  # endpoint hiccup — degrade gracefully
        return f"(failed to fetch claims: {exc})"
