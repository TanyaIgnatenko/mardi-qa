"""Stage-level metrics for one answered question.

Everything here is deterministic and computed from a saved trace, so a metric can
be redefined and the whole set recomputed without re-running the system or
spending a model call. Only two measures in the suite need a judge — whether the
gold fact is asserted, and whether the query is a sensible one for the task — and
those live in the harness, not here.

The split that matters throughout: what a query RETRIEVED and what reached the
MODEL are different sets, and so are what the model was given and what it CITED.
Collapsing any of those hides a failure that has its own fix.
"""

from __future__ import annotations

import re
from typing import Iterable

QID = re.compile(r"\bQ\d{2,9}\b")


# --------------------------------------------------------------------- utils ---

def _ratio(hit: int, total: int) -> float | None:
    """None, not zero, when the denominator is empty: 'nothing to find' and
    'found nothing' are different outcomes and must not average together."""
    return None if total == 0 else hit / total


def qids_in(text: str) -> set[str]:
    return set(QID.findall(text or ""))


def qids_in_rows(rows: Iterable[dict]) -> set[str]:
    found: set[str] = set()
    for row in rows:
        for value in row.values():
            found |= qids_in(str(value))
    return found


# ------------------------------------------------------------------ entities ---

def entity_metrics(trace_extracted: dict, linked_qids: list[str], gold: dict) -> dict:
    """Three separate numbers, because three different things go wrong.

    Extraction can miss a mention, extraction can invent one, and linking can
    resolve a correctly-extracted mention to the wrong item. We have seen all
    three: "bootstrap" extracted cleanly but linked to a paper, "Mergesort"
    linked to the generic concept instead of the MathAlgoDB entry.
    """
    required = set(gold.get("required_entities", []))
    permitted = set(gold.get("permitted_entities", []))
    # Linking resolves what the question REFERS TO, not what the answer contains.
    # Scoring it against the answer's entities was wrong and made the column
    # meaningless: for "which software solves MINLP" linking should find MINLP,
    # while the answer needs seventeen solvers, so recall read 0.00 on a question
    # whose linking was perfect. Fall back to required only for gold written
    # before the distinction existed.
    pivots = set(gold.get("pivot_entities") or required)
    linked = set(linked_qids)

    mentions = trace_extracted.get("entities", []) or []
    mention_texts = [m.get("text", "") if isinstance(m, dict) else str(m) for m in mentions]

    return {
        "mentions_extracted": len(mention_texts),
        "linked_count": len(linked),
        # Did linking reach what the question is about?
        "linking_recall": _ratio(len(linked & pivots), len(pivots)),
        # Of what it linked, how much belongs to this question at all?
        "linking_precision": _ratio(len(linked & (pivots | required | permitted)), len(linked)),
        "linked_unexpected": sorted(linked - pivots - required - permitted),
    }


# -------------------------------------------------------------------- sparql ---

NUMBER = re.compile(r"-?\d[\d,]*")


def _first_number(rows: list[dict]) -> int | None:
    """The single value a COUNT query returns."""
    for row in rows:
        for value in row.values():
            m = NUMBER.fullmatch(str(value).strip())
            if m:
                return int(m.group(0).replace(",", ""))
    return None


def sparql_metrics(queries: list[str], results: list, gold_rows: set[str] | None,
                   gold: dict | None = None) -> dict:
    """Execution accuracy, judged by the shape of the answer the question wants.

    One question has many correct queries — different variable names, different
    clause order, a semicolon chain instead of two triples — so the measure is
    what executing it returns, never its text. But "the same set of identifiers"
    only works for a bounded list. A COUNT query returns a number and no
    identifiers at all, so set comparison silently compares two empty sets and
    reports a match. An unbounded list returns its LIMIT, never the gold's 1438,
    so set comparison reports a miss on a perfectly good query. Three shapes,
    three comparisons.
    """
    gold = gold or {}
    shape = gold.get("answer_shape") or ("count" if "numeric_answer" in gold else "list")
    attempts = len(queries)
    got = [qids_in_rows(r.rows) if hasattr(r, "rows") else set() for r in results]

    empty = not got or not any(got)
    final = set() if empty else got[-1]
    first = set() if empty else got[0]
    rows = [r.rows for r in results if hasattr(r, "rows")]

    if shape == "count":
        # Compare the value, not the identifiers. A count query returns neither.
        expected = gold.get("numeric_answer")
        tol = gold.get("numeric_tolerance_pct", 10) / 100.0
        def near(n: int | None) -> bool | None:
            if expected is None or n is None:
                return None
            return abs(n - expected) <= max(1, abs(expected) * tol)
        got_first = _first_number(rows[0]) if rows else None
        got_final = _first_number(rows[-1]) if rows else None
        return {
            "attempts": attempts, "repair_fired": attempts > 1,
            "returned_nothing": not rows or all(not r for r in rows),
            "answer_shape": shape,
            "count_returned": got_final,
            "first_query_exact": near(got_first),
            "final_query_exact": near(got_final),
            "final_missing": [], "final_extra": [],
        }

    def matches(found: set[str]) -> bool | None:
        if gold_rows is None:
            return None
        if shape == "truncated_list":
            # The query cannot return all 1438 matches; what it returns must be a
            # subset of the gold, which is the most a capped listing can promise.
            return bool(found) and found <= gold_rows
        return found == gold_rows

    return {
        "attempts": attempts,
        "repair_fired": attempts > 1,
        # No rows at all is a definite miss, not an unmeasured case: reporting
        # None would hide a total failure behind "not applicable".
        "returned_nothing": empty,
        "answer_shape": shape,
        "first_query_exact": matches(first),
        "final_query_exact": matches(final),
        "final_missing": sorted(gold_rows - final) if gold_rows else [],
        "final_extra": sorted(final - gold_rows) if gold_rows else [],
    }


# ----------------------------------------------------- retrieval vs context ---

def qids_in_rows_or_labels(rows: Iterable[dict],
                           label_to_qid: dict[str, str] | None = None) -> set[str]:
    """Identifiers a result carries, whether as identifiers or as names.

    SPARQL returns only the variables the SELECT projects. A query that matched
    the right thirteen formulas and wrote `SELECT ?formulaLabel` returns their
    names and no identifiers at all — the binding existed inside the engine and
    was never asked for. Scored on identifiers alone that read as retrieval
    failure, 0.00, for a query that had found exactly the right rows.

    Retrieval is about whether the query found the right things, so it is
    measured here regardless of how the answer was projected. The cost of a
    missing identifier is real but belongs to the citation metrics: without one
    the answer cannot carry a portal link the reader can check.
    """
    found = qids_in_rows(rows)
    if not label_to_qid:
        return found
    for row in rows:
        for value in row.values():
            qid = label_to_qid.get(" ".join(str(value).split()).lower())
            if qid:
                found.add(qid)
    return found


def retrieval_metrics(results: list, context_rows: list[dict], gold: dict,
                      label_to_qid: dict[str, str] | None = None) -> dict:
    """Recall at two cut points, and the loss between them.

    retrieval_recall asks whether the query found the gold nodes. context_recall
    asks whether they survived into the prompt. The difference is damage done by
    truncation alone, and it has a different fix from a bad query: raise the cap
    or order better, rather than rewrite the query.
    """
    required = set(gold.get("required_entities", []))
    retrieved = set()
    for r in results:
        if hasattr(r, "rows"):
            retrieved |= qids_in_rows_or_labels(r.rows, label_to_qid)
    in_context = qids_in_rows_or_labels(context_rows, label_to_qid)

    retrieval_recall = _ratio(len(retrieved & required), len(required))
    context_recall = _ratio(len(in_context & required), len(required))
    loss = (None if retrieval_recall is None or context_recall is None
            else round(retrieval_recall - context_recall, 4))

    total_rows = sum(len(r.rows) for r in results if hasattr(r, "rows"))

    # Two cuts stand between what matches and what the model sees, and they need
    # separate numbers. The LIMIT cut comes first: 1438 publications match, the
    # query returns 50. The render cut comes second: of those 50, some number
    # reaches the prompt. Aligning the render cap to the query limit drove the
    # second to zero — which is why `coverage` matters and `truncation_loss`
    # alone stopped saying anything.
    true_total = gold.get("expected_counts", {}).get("publications")
    if true_total is None:
        true_total = gold.get("numeric_answer")
    coverage = (None if not true_total or total_rows == 0
                else round(min(1.0, total_rows / true_total), 4))

    return {
        "rows_retrieved": total_rows,
        "rows_in_context": len(context_rows),
        "true_total": true_total,
        # What fraction of the matching set the query actually brought back.
        "coverage": coverage,
        "retrieval_recall": retrieval_recall,
        "context_recall": context_recall,
        # > 0 means the query was right and the render cut threw the answer away.
        "truncation_loss": loss,
        "limit_truncated": bool(true_total and total_rows < true_total),
        "render_truncated": len(context_rows) < total_rows,
        # For a count question there are no gold identifiers, so both recalls
        # above are None and nothing checks the middle of the chain: the query
        # returned 1438 and the answer says 1438, but whether the number ever
        # reached the prompt went unverified. That gap produced a real defect —
        # an answer of "40 publications" where 1438 matched, the length of the
        # rendered list standing in for the total — and it was indistinguishable
        # from the model misreading data it had been given correctly.
        "count_in_context": _count_in_context(context_rows, gold),
    }


def _count_in_context(context_rows: list[dict], gold: dict) -> bool | None:
    """Did the gold number actually appear in what the model was shown?

    None for anything that is not a count question, so the column stays honest
    about which questions it speaks for.
    """
    if (gold.get("answer_shape") or "") != "count":
        return None
    expected = gold.get("numeric_answer")
    if expected is None:
        return None
    for row in context_rows:
        for value in row.values():
            if str(value).strip() == str(expected):
                return True
    return False


def schema_metrics(properties_shown: list[str], gold: dict) -> dict:
    """Was the model shown the properties its answer required?

    A query cannot use a relation the model was never told exists, so a low
    coverage here means the generation stage was set an impossible task and the
    query it wrote is not evidence about the model. Four defects this session
    were exactly this — P1463, P226, P1766, P1560/P1656 absent from the schema
    we passed — and each was found by reading a generated query by hand.
    """
    needed = set(re.findall(r"\bP\d+\b", gold.get("gold_sparql") or ""))
    if not needed:
        return {"properties_needed": [], "properties_missing": [],
                "schema_coverage": None}
    shown = set(properties_shown or ())
    missing = sorted(needed - shown)
    return {
        "properties_needed": sorted(needed),
        "properties_missing": missing,
        "schema_coverage": _ratio(len(needed & shown), len(needed)),
    }


# --------------------------------------------------------------------- answer ---

def source_metrics(answer: str, retrieved_qids: set[str], gold: dict,
                   linked_qids: Iterable[str] = ()) -> dict:
    """Four outcomes for cited identifiers, not two.

    A citation can be right, missing, superfluous or fabricated, and the last two
    are different failures: a superfluous link is noise, a fabricated one is a
    false statement. Our judge caught fabrication in only 5 of 27 cases, which is
    why this is deterministic.
    """
    cited = qids_in(answer)
    required = set(gold.get("required_entities", []))
    permitted = set(gold.get("permitted_entities", []))
    # Permitted entries may be prose ("any publication linked via ...") rather
    # than QIDs; those cannot be matched, so anything retrieved counts as allowed.
    permitted_qids = {p for p in permitted if QID.fullmatch(p)}
    # The entities the question was about belong here too. They are the subject,
    # not the result, so a query that asks "which publications used BayesDA"
    # returns publications and never BayesDA itself — and the closing link to
    # BayesDA's portal page was being scored as a fabricated identifier, the
    # worst category, for citing the very thing the user asked about.
    allowed = required | permitted_qids | retrieved_qids | set(linked_qids)

    # Entities the graph does link but that do not answer the question — AMPL
    # among MINLP solvers, say. Reported by name rather than lumped in with the
    # rest of the noise: citing one is a judgement failure, not a retrieval one.
    excluded = set(gold.get("excluded_from_sources", []))

    correct = cited & required
    # Surplus is what the gold neither requires nor permits. Subtracting only the
    # required set made `permitted_entities` decorative: the twelve authors a
    # question explicitly asks for, and the solvers that answer the first half of
    # another, were counted as extra facts while the gold named them as expected.
    superfluous = cited & allowed - required - permitted_qids - excluded - set(linked_qids)
    fabricated = cited - allowed - excluded

    return {
        "cited_count": len(cited),
        # Counts beside the ratio: "5/10" says how much was at stake, where 0.50
        # cannot tell one link missing out of two from fifty out of a hundred.
        "linked_correct": len(correct),
        "linked_required": len(required),
        "source_recall": _ratio(len(correct), len(required)),
        # Precision asks what share of the citations were legitimate, so the
        # numerator is everything allowed — not only the required set. Scoring it
        # against `required` alone makes precision structurally zero for a count
        # question, where nothing is required in the result and citing the entity
        # the question was about is nonetheless right.
        "source_precision": _ratio(len(cited & allowed), len(cited)),
        "missed": sorted(required - cited),
        "wrongly_cited": sorted(cited & excluded),
        # None, not an empty list, when the gold does not say what the answer
        # should contain. Without a required set every citation counts as
        # surplus, and the easy set duly reported five legitimate example
        # publications and the package the question was about as "extra facts".
        "superfluous": sorted(superfluous) if required else None,
        # Not in the gold and never retrieved — the answer cannot have got it
        # from the graph.
        "fabricated": sorted(fabricated),
    }


_NAME_STOP = {"the", "a", "an", "of", "for", "and", "or", "to", "in", "with", "on",
              "class", "algorithm", "software", "method", "using", "based",
              "from", "its", "it", "paper", "described", "graph", "data",
              "information", "available", "further", "about", "their", "this"}


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(w) > 2 and w not in _NAME_STOP}


def _naming_chunks(answer: str) -> list[str]:
    """The answer split into list items and paragraphs.

    Scope matters more than it sounds. A line-wide check called MAMSolver a
    mismatch for being named one line above its id; a 160-character window let a
    neighbouring bullet's words vouch for the wrong entity and hid five real
    mislabellings. A list item is the unit in which a model names one thing.
    """
    chunks, current = [], []
    for line in (answer or "").splitlines():
        starts_item = re.match(r"\s*([-*+]|\d+[.)])\s", line)
        if starts_item or not line.strip():
            if current:
                chunks.append("\n".join(current))
            current = [line] if line.strip() else []
        else:
            current.append(line)
    if current:
        chunks.append("\n".join(current))
    return chunks


def naming_metrics(answer: str, labels: dict[str, str]) -> dict:
    """Does the name the answer gives an identifier match the graph's own label?

    Fabricated identifiers were already caught; fabricated *names* for real ones
    were not. Asked which algorithms suit a sparse SPD system, an answer returned
    the eight nodes the graph links and presented six of them as MINRES, GMRES,
    Steepest Descent and Preconditioned CG, complete with convergence properties.
    Every identifier was genuine and retrieved, so nothing flagged it — but those
    six are category nodes labelled "Algorithm Class: Handles dense matrices" and
    the like, and the names were invented wholesale.

    `labels` comes from the graph, not from the run, because the defect appears
    precisely when the query forgot to ask for labels and the model filled the
    silence.
    """
    mislabelled = []
    for chunk in _naming_chunks(answer):
        body = re.sub(r"https?://\S+", " ", chunk)
        said = _content_words(QID.sub(" ", body))
        for qid in dict.fromkeys(QID.findall(body)):
            label = labels.get(qid)
            # An entity with no English label hands back its own id, leaving
            # nothing for the answer to contradict.
            if not label or label == qid:
                continue
            if not _content_words(label) & said:
                mislabelled.append({"qid": qid, "graph_label": label,
                                    "answer_said": " ".join(body.split())[:90]})
    checked = [q for q in dict.fromkeys(QID.findall(answer or ""))
               if labels.get(q) and labels[q] != q]
    return {
        "names_checked": len(checked),
        "mislabelled": mislabelled,
        "naming_fidelity": _ratio(len(checked) - len(mislabelled), len(checked)),
    }


def quote_fidelity(answer: str, context_text: str) -> dict:
    """Any quoted span must appear verbatim in what was retrieved.

    A misattributed quote is worse than a wrong identifier: it reads as evidence.
    Deterministic, so it does not depend on a judge noticing.
    """
    # Newlines excluded, and the length filter applied after matching rather than
    # inside the pattern. `"([^"]{20,300})"` skipped a short quoted phrase for
    # being under 20 characters and then paired its closing quote with the next
    # phrase's opening one, returning the markdown between them as the quote:
    # a link list and two line breaks reported as an unsupported citation.
    # Curly quotes count. A model that invents a property label writes it the way
    # prose is written — *“uses algorithm”* — and a pattern looking only for the
    # straight ASCII quote walked straight past exactly the defect this metric
    # exists to catch, in the same run where its straight-quoted twin was caught.
    # Floor of 12, not 20: an invented property label is short. "implements
    # algorithm" is exactly 20 and was caught, while "uses algorithm" is 14 and
    # walked past — the same fabrication, missed for being brief. Checked against
    # every answer in the set, 12 adds no false positives, and lowering it
    # further changes nothing, so the margin is real rather than tuned to taste.
    quotes = [q for q in re.findall(r'["“]([^"“”\n]{1,300})["”]', answer or "")
              if len(q) >= 12]
    haystack = " ".join((context_text or "").split())
    unsupported = [q for q in quotes if " ".join(q.split()) not in haystack]
    return {
        "quotes": len(quotes),
        "unsupported_quotes": unsupported,
        "quote_fidelity": _ratio(len(quotes) - len(unsupported), len(quotes)),
    }


def structural_validity(trace_extracted: dict, queries: list[str],
                        errors: list[str]) -> dict:
    """The gate before quality: can the model emit what the pipeline needs?

    A model that writes fluent prose but malformed JSON or broken SPARQL is
    unusable here regardless of how it scores elsewhere — one local model in an
    earlier comparison returned empty answers on every question.
    """
    syntax_errors = [e for e in errors if "malformed" in e.lower() or "400" in e]
    return {
        "extraction_parsed": bool(trace_extracted),
        "queries_generated": len(queries),
        "sparql_syntax_errors": len(syntax_errors),
    }
