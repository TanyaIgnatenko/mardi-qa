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

# How many linked entities get their schema shown to the generator. Must cover
# everything linking and keyword search between them produce, or the cap decides
# which candidate is allowed to be the query's subject — and it decided wrongly
# the moment a fourth candidate appeared.
SCHEMA_ENTITY_LIMIT = 6


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
    identifier_gloss: str = ""
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

    # An aggregate is never truncated: COUNT returns one row holding the whole
    # answer, so a trailing LIMIT 1 is decoration. Without this, such a query
    # satisfied "rows == limit" and was declared truncated, and the model dutifully
    # warned that more matches might exist — turning a complete, correct zero into
    # a hedged one.
    if re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", query, re.IGNORECASE):
        return ""

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
        aka = item.get("aka") or [] if isinstance(item, dict) else []
        if not mention:
            continue

        # Every name the thing goes by, through linking rather than through the
        # keyword fallback. A curator files MINLP under the label "MINLP" alone,
        # so "which packages solve mixed-integer nonlinear programs" matches no
        # label and linking finds nothing. The acronym did reach the graph, but
        # only as a topic keyword, and that path runs solely when linking comes
        # back empty — had linking resolved anything at all, the one term that
        # identifies the subject would never have been tried. Keyword search also
        # skips the type filter and the stub tie-break that linking applies.
        best = None
        for surface in [mention, *[a for a in aka if isinstance(a, str) and a.strip()]][:4]:
            try:
                best = pick_best_candidate(search_entities(surface), expected_kind=kind)
            except Exception:
                best = None
            if best:
                break
        if best:
            trace.linked_entities.append(best)

    # 2b. Nothing named? Search by the topic keywords instead ------------------
    # This is the path a described problem takes — "I have a large sparse SPD
    # system" names no entity. It used to be handled by telling the generator to
    # call the endpoint's full-text search service inside its SPARQL, which
    # returns HTTP 500 on this deployment: every such query failed, so the branch
    # was dead and described problems could not be answered at all.
    # Only when linking came back empty. For a while this ran unconditionally,
    # because a wrong link could switch it off — linking matched the acronym
    # "SPD" to an unrelated item labelled "spd" and the keyword search that had
    # been finding the actual problem class never ran. That is fixed where it
    # belonged: extraction now yields the full noun phrase and the right kind,
    # and label search finds "Problem Class: Symmetric positive definite (SPD)
    # linear systems" directly. Running it regardless cost one query per keyword,
    # up to six a question, on questions that no longer need it — enough extra
    # traffic to earn an HTTP 429 that fails the whole run rather than one query.
    if not trace.linked_entities:
        keywords = trace.extracted.get("keywords") or []
        if keywords:
            progress("No entity linked — searching by topic keywords…")
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
        # Covers every linked entity, not the first three. The cap silently
        # decided which candidate could be the query's subject: asked about a
        # sparse SPD system, the problem class landed fourth in the list and its
        # schema — the one carrying "solved by" — was never shown, so the query
        # could not be built from the entity the question was actually about.
        schema_text = format_schema(trace.linked_entities[:SCHEMA_ENTITY_LIMIT])
    else:
        # Nothing linked (topic questions, mwapi search): no entity schema exists,
        # so fall back to the verified core properties.
        schema_text = ("(no entities linked — use full-text search; core properties:)\n"
                       + _format_properties(CORE_PROPERTIES))

    # The core properties go in either way. An entity's own schema describes the
    # entity the question NAMES, never the class it ASKS FOR, and the properties
    # of that class are then invisible: asked which publications since 2020 used
    # SCIP for scheduling, the model saw SCIP's claims, found nothing resembling a
    # subject classification, dropped the scheduling condition, answered about
    # forty publications instead of three — and explained that the graph records
    # no problem-class field, which it does, as "Mathematics Subject
    # Classification ID" on the publications themselves.
    schema_text += ("\n\nProperties of the classes you may need to query "
                    "(verified against this endpoint):\n"
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
{_format_subject_codes(trace.extracted)}

Schema of these entities — the ONLY properties you may use. Direction matters:
a property listed under "→ X" has the entity as the subject, one listed under
"X →" has it as the object.

{schema_text}
"""
    query = sanitize_query(llm.complete(prompts.SPARQL_GENERATION_SYSTEM, gen_user))
    trace.queries.append(query)

    # 4. Execute, retry once on failure/empty/zero ------------------------------
    progress("Running SPARQL against the MaRDI endpoint…")
    result = _try_run(query, trace)

    # A count of zero is an empty answer wearing one row, and it was taking the
    # successful path: `{"n": "0"}` is not is_empty, so the claims were never
    # fetched. ANSWER_SYSTEM only permits asserting that something is absent when
    # the entity's actual claims are in evidence, so the model had no way to earn
    # that and hedged every true zero away — "0 results, but the relation may be
    # stored under a different property" — which is exactly the answer the user
    # cannot act on. Fetching the claims here lets a real zero be stated plainly
    # and a wrong-property zero still be caught.
    if result is None or result.is_empty or _is_zero_count(result):
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
        if (result is None or result.is_empty or _is_zero_count(result)) \
                and trace.linked_entities:
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
    # Recorded on the trace, not only appended to the prompt: anything the model
    # is shown has to be inside what the quote check verifies against, or quoting
    # it correctly counts as unsupported. Adding this block without recording it
    # did exactly that — two category labels, copied faithfully, flagged as
    # invented evidence.
    trace.identifier_gloss = _identify_bare_qids(trace.context_rows, progress)
    if trace.identifier_gloss:
        retrieved_parts.append("What those identifiers are:\n" + trace.identifier_gloss)

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


def _format_subject_codes(extracted: dict) -> str:
    """The MSC codes extraction found for the question's topics.

    Carried into the generation prompt because nothing else carries it. The
    extraction step was asked for these codes and duly returned "90B35" for
    scheduling, but the prompt listed only the question, the target type, the
    linked entities, the keywords and the schema — so the one value that could
    express "for scheduling problems" as a filter never reached the model, and
    the query came back without that condition for the third run in a row.
    """
    codes = []
    for item in extracted.get("entities", []) or []:
        if not isinstance(item, dict):
            continue
        msc = item.get("msc")
        for code in ([msc] if isinstance(msc, str) else (msc or [])):
            if isinstance(code, str) and code.strip():
                codes.append(f'{item.get("text", "?")} = MSC {code.strip()}')
    if not codes:
        return ""
    return ("Subject codes for the topics named in the question — filter with "
            "wdt:P226 and STRSTARTS to honour them:\n  " + "\n  ".join(codes))


_QID_IN_ROW = re.compile(r"^Q\d{2,9}$")


def _identify_bare_qids(rows: list[dict], progress) -> str:
    """Label, and characterise, every identifier that arrived without a name.

    The generation prompt asks for labels alongside IDs and the model does not
    always comply: asked which algorithms suit a sparse SPD system it selected
    `?algorithm` alone, received eight bare QIDs, and named six of them MINRES,
    GMRES, Steepest Descent and Preconditioned CG — inventions, since those six
    are category nodes labelled "Algorithm Class: Handles dense matrices" and the
    like. It could not have known: nothing in the rows said what they were.

    The implementation and source counts come along because they are what
    separates a method from a category. P31 does not: all eight are `algorithm`.
    """
    bare = {v for row in rows for v in row.values()
            if isinstance(v, str) and _QID_IN_ROW.match(v)}
    named = {v.rsplit("/", 1)[-1] for row in rows for v in row.values()
             if isinstance(v, str) and "/Q" in v}
    bare -= named
    if not bare:
        return ""

    progress("Looking up what the returned identifiers are…")
    query = f"""{config.SPARQL_PREFIXES}
SELECT ?x ?xLabel (COUNT(DISTINCT ?impl) AS ?implementations)
                  (COUNT(DISTINCT ?src) AS ?sources) WHERE {{
  VALUES ?x {{ {' '.join('wd:' + q for q in sorted(bare)[:60])} }}
  OPTIONAL {{ ?x wdt:P1766 ?impl . }}
  OPTIONAL {{ ?x wdt:P286 ?src . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}} GROUP BY ?x ?xLabel"""
    try:
        rows_out = run_query(query).rows
    except SparqlError:
        return ""

    lines = []
    for r in rows_out:
        qid = r.get("x", "").rsplit("/", 1)[-1]
        label = r.get("xLabel") or "(no label in the graph)"
        impls, srcs = r.get("implementations", "0"), r.get("sources", "0")
        lines.append(f"{qid} | {label} | implementations: {impls} | sources: {srcs}")
    return "\n".join(sorted(lines))


def _is_zero_count(result: SparqlResult | None) -> bool:
    """A single row whose only value is 0 — an aggregate that matched nothing."""
    if result is None or len(result.rows) != 1:
        return False
    values = list(result.rows[0].values())
    return len(values) == 1 and str(values[0]).strip() in ("0", "0.0")


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
