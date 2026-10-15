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

import re
from dataclasses import dataclass, field

from . import config, prompts
from .entity_linking import (
    CORE_PROPERTIES,
    LinkedEntity,
    describe_entity,
    format_schema,
    pick_best_candidate,
    search_by_keywords,
    search_entities,
)
from .llm import LLM
from .sparql_client import (
    SparqlError,
    SparqlResult,
    query_limit,
    run_query,
    sanitize_query,
    to_count_query,
)


@dataclass
class PipelineTrace:
    """Everything the UI needs to show the user what happened."""
    route: str = "sparql"          # which gateway route produced this answer
    route_reason: str = ""         # classifier's one-line justification
    route_source: str = ""         # "llm" | "heuristic" | "" (router disabled)
    extracted: dict = field(default_factory=dict)
    linked_entities: list[LinkedEntity] = field(default_factory=list)
    properties_shown: list[str] = field(default_factory=list)
    schema_text: str = ""
    queries: list[str] = field(default_factory=list)
    query_errors: list[str] = field(default_factory=list)
    # Rows that actually reached the answer prompt. A query capped at LIMIT 50
    # whose rendering caps at 40 silently drops ten, so "retrieved" and "seen by
    # the model" are different sets — and an answer can be missing a fact the
    # query found. Recording both is what makes that loss measurable.
    context_rows: list[dict] = field(default_factory=list)
    # QIDs that came from topic-keyword search rather than from a named mention.
    # Kept apart so linking precision is not judged on candidates the question
    # never named.
    keyword_candidates: list[str] = field(default_factory=list)
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


def _truncation_note(query: str, result: SparqlResult, progress) -> str:
    """Tell the answer step the real total whenever a listing hit its LIMIT.

    A query capped at LIMIT 50 returns 50 rows whether 51 or 1438 match, and the
    row count then gets reported as the answer — "40 publications used BayesDA"
    when 1438 did. One extra COUNT query removes the guesswork; when the query
    cannot be rewritten, say the total is unknown rather than let it be inferred.
    """
    limit = query_limit(query)
    if limit is None or len(result.rows) < limit:
        return ""  # everything that matched came back

    counter = to_count_query(query)
    if counter:
        progress("Result hit the row limit — counting the full set…")
        try:
            counted = run_query(counter)
            if counted.rows:
                total = next(iter(counted.rows[0].values()))
                return (f"TRUNCATED: this query was capped at LIMIT {limit}. The full "
                        f"number of matches is {total}. The rows above are a sample — "
                        f"state {total} as the total, never the number of rows shown.")
        except SparqlError:
            pass
    return (f"TRUNCATED: this query was capped at LIMIT {limit} and returned exactly "
            f"{limit} rows, so more matches exist and the true total is UNKNOWN. Present "
            f"the rows as examples. Do NOT state any total, and do NOT present the number "
            f"of rows shown as a count.")


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
    # Entities may arrive as bare strings (old format) or as {text, kind}. The
    # kind matters: "bootstrap" and "Mergesort" are both an R package and a
    # mathematical concept in MaRDI, and without a type to aim at the choice
    # degrades to "take the first search hit".
    progress("Linking entities to MaRDI items…")
    for item in trace.extracted.get("entities", [])[:5]:
        mention = item.get("text", "") if isinstance(item, dict) else str(item)
        kind = item.get("kind") if isinstance(item, dict) else None
        if not mention:
            continue
        try:
            best = pick_best_candidate(search_entities(mention), expected_kind=kind)
        except Exception:
            best = None
        if best:
            trace.linked_entities.append(best)

    # 2b. Nothing named? Search by the topic keywords instead ------------------
    # This is the path a described problem takes — "I have a large sparse SPD
    # system" names no entity. It used to be handled by telling the generator to
    # call the endpoint's full-text search service inside its SPARQL, which
    # returns HTTP 500 on this deployment: every such query failed, so the branch
    # was dead and described problems could not be answered at all.
    if not trace.linked_entities:
        keywords = trace.extracted.get("keywords") or []
        if keywords:
            progress("No entity named — searching by topic keywords…")
            candidates = search_by_keywords(keywords, limit=3)
            trace.keyword_candidates = [c.qid for c in candidates]
            trace.linked_entities.extend(candidates)

    # 3. Generate SPARQL ---------------------------------------------------------
    # The schema comes from the linked entities themselves, not from scoring the
    # global catalog against the question's words. That scoring is what hid
    # P1463 ("describes a project that uses") from every publication question:
    # its label shares no token with "which publications used X", so it scored
    # zero out of 2,610 properties and the model invented one instead.
    progress("Loading the schema of the linked entities…")
    if trace.linked_entities:
        schema_text = format_schema(trace.linked_entities[:3])
    else:
        # Nothing linked (topic questions, mwapi search): no entity schema exists,
        # so fall back to the verified core properties.
        schema_text = ("(no entities linked — use full-text search; core properties:)\n"
                       + _format_properties(CORE_PROPERTIES))

    # Record the properties the model was actually offered. Both sources count:
    # the entity schema built above AND the system prompt, which names a handful
    # of its own. Reading only the schema block reported P1463 missing for the
    # Mergesort question — the entity has no incoming P1463, which is why the
    # answer is zero — while the system prompt had offered it all along and the
    # model used it correctly. A metric that cries missing when nothing is
    # missing is worse than no metric.
    #
    # Four defects this session were genuinely this shape — P1463, P226, P1766,
    # P1560/P1656 absent from everything we showed — and each took reading a
    # generated query by hand to find.
    trace.properties_shown = sorted(set(
        re.findall(r"\bP\d+\b", schema_text)
        + re.findall(r"\bP\d+\b", prompts.SPARQL_GENERATION_SYSTEM)
    ))
    # Kept because the quote check needs it: property labels like "implements
    # algorithm" are quotable material the model was shown, but they come from
    # here rather than from the result rows, so checking quotes against the rows
    # alone reported a correctly quoted label as unsupported evidence.
    trace.schema_text = schema_text[:8000]

    progress("Generating SPARQL query…")
    gen_user = f"""Question: {question}
Answer should be a: {trace.extracted.get("target_type") or "(unspecified)"}

Linked MaRDI entities:
{_format_entities(trace.linked_entities)}

Topic keywords (for mwapi full-text search, if needed): {trace.extracted.get("keywords", [])}

Schema of these entities — the ONLY properties you may use. Direction matters:
a property listed under "→ X" has the entity as the subject, one listed under
"X →" has it as the object.

{schema_text}
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
        note = _truncation_note(q, r, progress)
        trace.context_rows.extend(r.rows[: config.CONTEXT_ROWS_IN_PROMPT])
        retrieved_parts.append(
            f"SPARQL query:\n{q}\n\nResults:\n{r.to_prompt_text(config.CONTEXT_ROWS_IN_PROMPT)}"
            + (f"\n\n{note}" if note else "")
        )
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
