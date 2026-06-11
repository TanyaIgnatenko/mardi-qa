"""Entity linking + schema discovery against the MaRDI Wikibase.

Key idea: we never hardcode MaRDI property IDs. Instead we
  1. resolve entity mentions ("ggplot2") to QIDs via wbsearchentities,
  2. fetch the full property catalog (PID -> label) once and cache it,
  3. fetch all truthy claims of linked entities (with labels) so the LLM
     sees the *actual* local schema around the entities in question.
"""

from __future__ import annotations

from dataclasses import dataclass

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


def pick_best_candidate(candidates: list[LinkedEntity]) -> LinkedEntity | None:
    """Prefer R/CRAN package items when the description hints at it."""
    if not candidates:
        return None
    for c in candidates:
        desc = (c.description or "").lower()
        if "r package" in desc or "cran" in desc:
            return c
    return candidates[0]


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
    return {pid: label for _, pid, label in scored[:top_k]}


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
