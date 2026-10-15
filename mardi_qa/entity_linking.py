"""Entity linking + schema discovery against the MaRDI Wikibase.

Key idea: we never hardcode MaRDI property IDs. Instead we
  1. resolve entity mentions ("ggplot2") to QIDs via wbsearchentities,
  2. fetch the full property catalog (PID -> label) once and cache it,
  3. fetch all truthy claims of linked entities (with labels) so the LLM
     sees the *actual* local schema around the entities in question.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import requests

from . import config
from .sparql_client import run_query


@dataclass
class LinkedEntity:
    mention: str
    qid: str
    label: str
    description: str

    def portal_url(self) -> str:
        return f"https://portal.mardi4nfdi.de/wiki/Item:{self.qid}"


def search_entities(mention: str, limit: int = 5) -> list[LinkedEntity]:
    """Resolve a surface form to candidate MaRDI items via wbsearchentities."""
    resp = requests.get(
        config.MEDIAWIKI_API,
        params={
            "action": "wbsearchentities",
            "search": mention,
            "language": "en",
            "type": "item",
            "limit": limit,
            "format": "json",
        },
        headers={"User-Agent": config.USER_AGENT},
        timeout=30,
    )
    resp.raise_for_status()
    out = []
    for hit in resp.json().get("search", []):
        out.append(
            LinkedEntity(
                mention=mention,
                qid=hit["id"],
                label=hit.get("label", ""),
                description=hit.get("description", ""),
            )
        )
    return out


def candidate_facts(qids: list[str]) -> dict[str, dict]:
    """Type and claim count for several candidates in ONE query.

    wbsearchentities returns only a label and a description, so the candidate's
    actual type is not available at link time — yet it is exactly what tells a
    package apart from a paper of the same name. One batched query is cheap
    enough to run before every choice.
    """
    if not qids:
        return {}
    values = " ".join(f"wd:{q}" for q in qids)
    try:
        result = run_query(f"""{config.SPARQL_PREFIXES}
SELECT ?x ?tLabel ?profLabel (COUNT(?v) AS ?claims) WHERE {{
  VALUES ?x {{ {values} }}
  OPTIONAL {{ ?x wdt:P31 ?t }}
  OPTIONAL {{ ?x wdt:P1460 ?prof }}
  OPTIONAL {{ ?x ?dp ?v }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}} GROUP BY ?x ?tLabel ?profLabel""")
    except Exception:
        return {}
    facts: dict[str, dict] = {}
    for row in result.rows:
        qid = row.get("x", "").rsplit("/", 1)[-1]
        entry = facts.setdefault(qid, {"types": set(), "profile": "", "claims": 0})
        if row.get("tLabel"):
            entry["types"].add(row["tLabel"].lower())
        if row.get("profLabel"):
            entry["profile"] = row["profLabel"].lower()
        entry["claims"] = max(entry["claims"], int(row.get("claims") or 0))
    return facts


def pick_best_candidate(
    candidates: list[LinkedEntity], expected_kind: str | None = None
) -> LinkedEntity | None:
    """Choose the item a mention refers to.

    The old rule scanned descriptions for "r package"/"cran". That fires only
    for packages: for a person, an algorithm or a paper it never matches and the
    choice silently degrades to "take the first search hit" — which is how
    "Mergesort" resolved to the generic merge-sort item instead of the MathAlgoDB
    one, and how a paper title resolved to a duplicate record.

    Two signals replace it:
      1. the candidate's actual type vs the type the question implies,
      2. claim count as a tie-breaker — MaRDI holds duplicate stubs (five items
         labelled "Yihui Xie", four with no description at all), and the stub is
         never the one the user means.
    """
    if not candidates:
        return None

    facts = candidate_facts([c.qid for c in candidates])

    # A candidate whose profile contradicts the kind asked for is excluded, not
    # merely down-ranked — and if that leaves nothing, we return None rather than
    # the best of a wrong set.
    #
    # Returning something wrong is worse than returning nothing, because nothing
    # lets the caller fall through to keyword search. Asked for the task "sparse
    # system", the search API offers the paper "Sparse system identification",
    # whose title matches the phrase exactly; ranked against publications only,
    # that paper wins and the query is then built on a citation relation. With it
    # excluded, keyword search runs instead and finds the actual problem class.
    kind = (expected_kind or "").lower()
    if kind in PROFILES:
        on_kind = [c for c in candidates
                   if kind in facts.get(c.qid, {}).get("profile", "")]
        if on_kind:
            candidates = on_kind
        else:
            # Nothing of the expected kind. An exact label match is still enough
            # evidence to accept the item anyway: MINLP is a class of problem in
            # mathematics but a software item in MaRDI, and the user naming it
            # exactly means that item whatever profile it was filed under. A
            # descriptive fragment carries no such evidence — "sparse system"
            # merely resembles the paper "Sparse system identification" — so
            # without an exact match we return nothing and let keyword search run.
            # Publications are excluded from this escape hatch unless one was
            # what the question asked for. There are 5.2 million of them and
            # their titles are arbitrary phrases, so "scheduling problems"
            # matching the paper titled "Scheduling problems" is coincidence; the
            # same match on a software or task label is identification.
            exact = [c for c in candidates
                     if c.label.strip().lower() == (c.mention or "").strip().lower()
                     and (kind == "publication"
                          or "publication" not in facts.get(c.qid, {}).get("profile", ""))]
            if not exact:
                return None
            candidates = exact

    # Words that mark a candidate's type as matching the expected kind. Kept
    # deliberately loose: MaRDI types are free-form labels, not a closed set.
    # Mirrors the kinds the extraction step can return, which in turn mirror the
    # graph's profile types. A kind with no entry here falls through to the
    # profile check alone, which is the reliable signal anyway.
    kind_words = {
        "task": ("problem", "task", "problem class"),
        "algorithm": ("algorithm", "method"),
        "model": ("model",),
        "software": ("package", "software", "library"),
        "publication": ("article", "publication", "paper", "journal"),
        "person": ("human", "person"),
        "dataset": ("data set", "dataset"),
        "formula": ("formula", "equation"),
        "concept": ("concept",),
    }.get((expected_kind or "").lower(), ())

    def score(c: LinkedEntity) -> tuple[int, int, int]:
        f = facts.get(c.qid, {})
        haystack = " ".join(f.get("types", set())) + " " + (c.description or "").lower()

        # MaRDI profile type is the decisive signal: a closed vocabulary that
        # matches our kinds exactly, and only curated MaRDI items carry it.
        # instance-of does not discriminate here — asked for an algorithm, both
        # "Mergesort" (profile: algorithm, 22 claims) and the imported generic
        # "merge sort" (types: "stable sorting algorithm", 86 claims) match the
        # word, and the claim count then picks the wrong one.
        profile_hit = 1 if (expected_kind and expected_kind.lower() in f.get("profile", "")) else 0
        type_hit = 1 if kind_words and any(w in haystack for w in kind_words) else 0
        if not kind_words and ("r package" in haystack or "cran" in haystack):
            type_hit = 1  # no expected kind: keep the old package preference

        # An exact label match beats everything except the profile. Without it,
        # richness alone prefers whatever is famous: asked for "SCIP" the scorer
        # returned SciPy, and asked for "MINLP" it returned MINLPLib — both much
        # larger records whose names merely resemble the mention.
        exact = 1 if c.label.strip().lower() == (c.mention or "").strip().lower() else 0

        # Claim count then separates the duplicates an exact match leaves tied:
        # "SCIP" and "scip" are both exact, and the real item carries 141 claims
        # against the stub's 42. It sits above the instance-of substring because
        # profile type is the reliable classifier and that substring is noise —
        # the stub is typed "r package" while the real SCIP is typed not at all.
        return (profile_hit, exact, f.get("claims", 0), type_hit)

    # max() keeps the first maximum, which preserves search-rank order on ties.
    return max(candidates, key=score)


def fetch_property_catalog() -> dict[str, str]:
    """All properties in the MaRDI KG as {PID: english label}. Cache upstream."""
    query = f"""{config.SPARQL_PREFIXES}
SELECT ?p ?pLabel WHERE {{
  ?p a wikibase:Property .
  ?p rdfs:label ?pLabel .
  FILTER(LANG(?pLabel) = "en")
}}
LIMIT 5000
"""
    result = run_query(query)
    catalog = {}
    for row in result.rows:
        pid = row["p"].rsplit("/", 1)[-1]
        catalog[pid] = row["pLabel"]
    return catalog


# Properties verified against the live endpoint. The lexical scoring below cannot
# find these on its own — "describes a project that uses" shares no token with a
# question like "which publications used X", so P1463 never reached the prompt and
# the model invented P4 / P1988 / P4317 instead. Anything load-bearing belongs
# here, not in the scoring.
CORE_PROPERTIES: dict[str, str] = {
    "P1463": "describes a project that uses — publication -> software/dataset it uses "
             "(USE THIS for 'which publications used X' and counts of them)",
    "P19":   "maintained by — software -> person",
    "P16":   "author — software or publication -> person",
    "P585":  "imports — software -> software",
    "P342":  "depends on software — software -> software",
    "P28":   "publication date — publication -> date (use YEAR(?d) to filter)",
    "P31":   "instance of — e.g. wd:Q57080 R package, wd:Q56887 scholarly article",
    # Publications carry MSC codes and nothing else advertises them: a question
    # filtering by subject area ("SCIP used for scheduling") needs this, and
    # without it the topic condition is silently dropped while the date filter —
    # which IS listed here — gets applied. Filter with STRSTARTS on the code
    # prefix, e.g. STRSTARTS(?msc, "90B35") for scheduling.
    "P226":  "Mathematics Subject Classification ID — publication -> MSC code "
             "string, the only way to filter publications by subject area",
    "P159":  "title — publication -> string",
    "P200":  "published in — publication -> journal",
    "P1458": "similar software — software -> software",
    # Algorithms link to the software implementing them and to the publications
    # describing them, and neither relation is reachable any other way. Without
    # these two, "which software implements LU decomposition" cannot be answered
    # at all: every model asked it either invented a property or produced
    # malformed SPARQL, because the one it needed was not on offer.
    "P1766": "implemented by — algorithm -> software that implements it",
    "P286":  "described by source — algorithm/model -> publication describing it "
             "(distinct from P1463, which is a publication USING software)",
    "P1578": "solved by — problem class -> algorithms that solve it",
    "P1560": "contains — model -> the formulas and conditions it is made of",
    "P1656": "discretized by — model -> its discretization scheme",
    "P43":   "author name string — literal, NOT a linked person",
}


def relevant_properties(
    catalog: dict[str, str], question: str, top_k: int = config.MAX_PROPERTIES_IN_PROMPT
) -> dict[str, str]:
    """Cheap lexical retrieval of properties likely relevant to the question,
    always including a core set that matters for software/package questions."""
    core_keywords = (
        "instance of", "depend", "maintain", "author", "developer", "description",
        "programmed in", "software", "license", "version", "cites", "cited",
        "repository", "url", "doi", "subclass", "part of", "publication",
        "imports", "title", "keyword", "topic", "field",
    )
    q_tokens = {t.strip("?.,!\"'()").lower() for t in question.split() if len(t) > 3}

    scored: list[tuple[int, str, str]] = []
    for pid, label in catalog.items():
        ll = label.lower()
        score = 0
        if any(kw in ll for kw in core_keywords):
            score += 2
        if any(tok in ll for tok in q_tokens):
            score += 3
        if score:
            scored.append((score, pid, label))
    scored.sort(key=lambda x: (-x[0], x[1]))
    picked = {pid: label for _, pid, label in scored[:top_k]}

    # Core properties always win a slot: they are the ones the query actually
    # needs, and losing them to lexical scoring is how the model ends up guessing.
    return {**picked, **CORE_PROPERTIES}


# ------------------------------------------------- keyword / topic search ---

# MaRDI profile types. These are the curated classification of the graph and the
# only reliable way to keep a label search off the 5.2M publications, which
# otherwise fill every result: an unscoped search for "positive definite" returns
# papers with those words in the title, never the problem class of that name.
PROFILES = {
    "task": "Q6534272",
    "algorithm": "Q6503323",
    "model": "Q6534270",
    "software": "Q5976450",
    "dataset": "Q5984635",
    "formula": "Q5981696",
    "concept": "Q7283889",
    "publication": "Q5976449",
    "person": "Q5976445",
}

# Searched when the question describes something instead of naming it. Omits
# publications and people on purpose: those are the two largest profiles and
# neither is ever the pivot of a "what should I use for this problem" question.
TOPIC_PROFILES = ("task", "algorithm", "model", "software", "concept", "formula")


def search_by_keywords(
    keywords: list[str], profiles: tuple[str, ...] = TOPIC_PROFILES, limit: int = 12
) -> list[LinkedEntity]:
    """Find entities whose label contains one of the keywords.

    Replaces the endpoint's full-text search service, which returns HTTP 500 on
    this deployment — every query the generator built with it failed, so the whole
    "nothing linked, fall back to searching" path was dead. A label substring
    match is cruder and unranked, but it works, and scoped to the non-publication
    profiles it finds the entity a described problem refers to.
    """
    terms = [k.strip().lower() for k in keywords if len(k.strip()) > 3][:6]
    if not terms:
        return []

    profile_values = " ".join(f"wd:{PROFILES[p]}" for p in profiles if p in PROFILES)

    # One query per term, most specific term first, rather than a single OR.
    # A disjunction lets a common word swamp the window: "symmetric" matches
    # hundreds of package names, the endpoint returns an arbitrary slice of them,
    # and "positive definite" — the term that identifies the problem class — never
    # appears in it. Giving each term its own window fixes that; the queries are
    # small and scoped, so six of them still cost a couple of seconds.
    found: dict[str, LinkedEntity] = {}
    for term in sorted(terms, key=len, reverse=True):
        query = f"""{config.SPARQL_PREFIXES}
SELECT DISTINCT ?x ?l ?desc WHERE {{
  VALUES ?prof {{ {profile_values} }}
  ?x wdt:P1460 ?prof ; rdfs:label ?l .
  FILTER(LANG(?l) = "en" && CONTAINS(LCASE(?l), "{term}"))
  OPTIONAL {{ ?x schema:description ?desc . FILTER(LANG(?desc) = "en") }}
}}
LIMIT {limit * 2}"""
        try:
            result = run_query(query)
        except Exception:
            continue
        for row in result.rows:
            qid = row.get("x", "").rsplit("/", 1)[-1]
            if qid and qid not in found:
                found[qid] = LinkedEntity(mention=term, qid=qid,
                                          label=row.get("l", ""),
                                          description=row.get("desc", ""))

    # Rank by how much of the question each label covers: a label matching three
    # of the terms describes the asked-about thing far better than one matching a
    # single common word.
    def score(e: LinkedEntity) -> tuple[int, int]:
        label = e.label.lower()
        matched = [t for t in terms if t in label]
        return (len(matched), max((len(t) for t in matched), default=0))

    return sorted(found.values(), key=lambda e: (-score(e)[0], -score(e)[1], e.qid))[:limit]


# ------------------------------------------------------------------ schema ---

# Cap on the triples scanned when asking which properties point AT an entity.
# Hub items are referenced by tens of thousands of others — "programmed in" alone
# links ~20k packages to R — and an unbounded scan times the endpoint out. The
# cost of the cap is that on a hub the incoming list is a lower bound.
INCOMING_SCAN_LIMIT = 20000


@lru_cache(maxsize=512)
def entity_schema(qid: str) -> dict:
    """The properties an entity actually participates in, both directions.

    Returns {"out": [...], "in": [...]} where each item is
    {"pid", "label", "type"} — `type` being the class at the *other* end.

    Both directions are required, and this is the whole point of the function.
    A statement lives on its subject: "publication P1463 software" is a claim of
    the PUBLICATION, so glmnet's own claims contain no P1463 at all. Reading only
    outgoing edges — which is what describe_entity does — makes the property
    that answers "which publications used glmnet" invisible, and the model then
    invents one. External identifiers (VIAF, ORCID, Twitter…) are dropped: they
    are never query targets and on a person they are over half the properties.
    """
    return {"out": _schema_side(qid, incoming=False),
            "in": _schema_side(qid, incoming=True)}


def _schema_side(qid: str, incoming: bool) -> list[dict]:
    if incoming:
        # LIMIT sits in the subquery so it bounds the scan, not the result.
        # The subject's type must be OPTIONAL: joining it as a required pattern
        # across the scanned subjects times the endpoint out (measured — the
        # required form dies, the optional form returns in seconds).
        pattern = (f"{{ SELECT ?dp ?s WHERE {{ ?s ?dp wd:{qid} }} "
                   f"LIMIT {INCOMING_SCAN_LIMIT} }}\n  OPTIONAL {{ ?s wdt:P31 ?t }}")
    else:
        # OPTIONAL: dates, strings and URLs have no instance-of, and without it
        # those properties would drop out of the result entirely.
        pattern = f"wd:{qid} ?dp ?v .\n  OPTIONAL {{ ?v wdt:P31 ?t }}"

    query = f"""{config.SPARQL_PREFIXES}
SELECT DISTINCT ?prop ?propLabel ?tLabel WHERE {{
  {pattern}
  ?prop wikibase:directClaim ?dp ; wikibase:propertyType ?pt .
  FILTER(?pt != wikibase:ExternalId)
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}"""
    try:
        result = run_query(query)
    except Exception:
        return []  # caller falls back to CORE_PROPERTIES

    merged: dict[str, dict] = {}
    for row in result.rows:
        pid = row.get("prop", "").rsplit("/", 1)[-1]
        if not pid:
            continue
        entry = merged.setdefault(
            pid, {"pid": pid, "label": row.get("propLabel", pid), "types": []}
        )
        t = row.get("tLabel")
        if t and t not in entry["types"]:
            entry["types"].append(t)
    return sorted(merged.values(), key=lambda e: e["pid"])


def format_schema(entities: list[LinkedEntity]) -> str:
    """Render the schema of the linked entities for the generation prompt.

    Direction is spelled out rather than implied. It has to be: P585 appears on
    both sides for a package — "what glmnet imports" and "what imports glmnet"
    are different triples on the same property, and only the position separates
    them.
    """
    if not entities:
        return "(no entities linked — no entity schema available)"

    blocks = []
    for e in entities:
        schema = entity_schema(e.qid)
        lines = [f'{e.label} ({e.qid}):']
        for side, arrow, caption in (
            ("out", f"{e.label} → X", "this entity is the SUBJECT"),
            ("in", f"X → {e.label}", "this entity is the OBJECT"),
        ):
            props = schema.get(side, [])
            if not props:
                continue
            lines.append(f"  {arrow}  ({caption}):")
            for p in props:
                types = f" → {', '.join(p['types'][:3])}" if p["types"] else ""
                lines.append(f"    wdt:{p['pid']}  {p['label']}{types}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def describe_entity(qid: str, max_rows: int = 120) -> str:
    """All truthy claims of an entity with human-readable labels.

    Used both to show the LLM the local schema and as fallback context
    when a generated query returns nothing.
    """
    query = f"""{config.SPARQL_PREFIXES}
SELECT ?propLabel ?prop ?valueLabel ?value WHERE {{
  wd:{qid} ?directProp ?value .
  ?prop wikibase:directClaim ?directProp .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT {max_rows}
"""
    result = run_query(query)
    if result.is_empty:
        return f"(no claims found for {qid})"
    lines = []
    for row in result.rows:
        pid = row.get("prop", "").rsplit("/", 1)[-1]
        lines.append(
            f"{row.get('propLabel', pid)} [{pid}]: "
            f"{row.get('valueLabel') or row.get('value', '')}"
        )
    return "\n".join(lines)
