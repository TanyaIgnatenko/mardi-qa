"""Thin SPARQL client for the MaRDI Wikibase endpoint."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field

import requests

from . import config


class SparqlError(Exception):
    """Raised when the endpoint rejects or fails a query."""


_DISCOVERED_ENDPOINT: str | None = None


def discover_endpoint() -> str:
    """Return a working SPARQL endpoint URL.

    Uses MARDI_SPARQL_ENDPOINT if set; otherwise probes the candidate paths
    with a trivial query and caches the first one that answers correctly.
    """
    global _DISCOVERED_ENDPOINT
    if config.SPARQL_ENDPOINT:
        return config.SPARQL_ENDPOINT
    if _DISCOVERED_ENDPOINT:
        return _DISCOVERED_ENDPOINT

    probe = "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"
    errors = []
    for url in config.SPARQL_ENDPOINT_CANDIDATES:
        try:
            resp = requests.get(
                url,
                params={"query": probe, "format": "json"},
                headers={
                    "Accept": "application/sparql-results+json",
                    "User-Agent": config.USER_AGENT,
                },
                timeout=20,
            )
            if resp.status_code == 200 and "results" in resp.json():
                _DISCOVERED_ENDPOINT = url
                return url
            errors.append(f"{url} -> HTTP {resp.status_code}")
        except Exception as exc:
            errors.append(f"{url} -> {exc}")
    raise SparqlError(
        "No working SPARQL endpoint found. Tried:\n" + "\n".join(errors)
        + "\nFind the real one: open https://query.portal.mardi4nfdi.de, run any "
        "query, and check the request URL in browser DevTools (Network tab). "
        "Then set MARDI_SPARQL_ENDPOINT to that URL."
    )


@dataclass
class SparqlResult:
    columns: list[str] = field(default_factory=list)
    rows: list[dict[str, str]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.rows) == 0

    def to_prompt_text(self, max_rows: int = 40) -> str:
        """Compact, LLM-friendly rendering of the result table."""
        if self.is_empty:
            return "(no results)"
        lines = [" | ".join(self.columns)]
        for row in self.rows[:max_rows]:
            lines.append(" | ".join(row.get(c, "") for c in self.columns))
        if len(self.rows) > max_rows:
            lines.append(f"... ({len(self.rows) - max_rows} more rows truncated)")
        return "\n".join(lines)


def sanitize_query(query: str) -> str:
    """Strip markdown fences, ensure prefixes and a LIMIT clause."""
    query = query.strip()
    query = re.sub(r"^```(?:sparql)?\s*", "", query)
    query = re.sub(r"\s*```$", "", query)
    query = query.strip()

    # Prepend standard prefixes for any that the query uses but does not declare.
    declared = {m.lower() for m in re.findall(r"PREFIX\s+(\w+):", query, re.I)}
    needed_lines = []
    for line in config.SPARQL_PREFIXES.splitlines():
        m = re.match(r"PREFIX\s+(\w+):", line)
        if m and m.group(1).lower() not in declared:
            needed_lines.append(line)
    if needed_lines:
        query = "\n".join(needed_lines) + "\n" + query

    # Inject a LIMIT if the query is a SELECT without one (protect the endpoint).
    if re.search(r"\bSELECT\b", query, re.I) and not re.search(r"\bLIMIT\s+\d+", query, re.I):
        query = query.rstrip().rstrip(";") + f"\nLIMIT {config.DEFAULT_RESULT_LIMIT}"
    return query


def query_limit(query: str) -> int | None:
    """The LIMIT a query carries, if any."""
    m = re.search(r"\bLIMIT\s+(\d+)", query, re.I)
    return int(m.group(1)) if m else None


def to_count_query(query: str) -> str | None:
    """Rewrite a listing SELECT into a COUNT of its first variable.

    A listing query capped at LIMIT 50 returns 50 rows whether the real total is
    51 or 1438, and the answer step has no way to tell — which is how "40
    publications" got reported for a package used by 1438. Asking the endpoint
    for the count is the only way to know.

    Conservative on purpose: aggregates, sub-selects and anything whose shape we
    cannot rewrite safely return None, and the caller then says the total is
    unknown instead of guessing.
    """
    if re.search(r"\bCOUNT\s*\(", query, re.I) or re.search(r"\bGROUP\s+BY\b", query, re.I):
        return None  # already an aggregate
    if query.count("SELECT") > 1:
        return None  # sub-select: the rewrite would target the wrong clause

    m = re.search(r"\bSELECT\b(\s+DISTINCT)?\s+(.+?)\s*\bWHERE\b", query, re.I | re.S)
    if not m:
        return None
    variables = re.findall(r"\?(\w+)", m.group(2))
    if not variables:
        return None  # SELECT * — no single thing to count

    body = query[m.end():]
    body = re.sub(r"\bLIMIT\s+\d+", "", body, flags=re.I)
    body = re.sub(r"\bOFFSET\s+\d+", "", body, flags=re.I)
    body = re.sub(r"\bORDER\s+BY\b.+?(?=\}|$)", "", body, flags=re.I | re.S)
    prefixes = query[: m.start()]
    return (f"{prefixes}SELECT (COUNT(DISTINCT ?{variables[0]}) AS ?total_count) "
            f"WHERE{body}")


def run_query(query: str, timeout: int = config.SPARQL_TIMEOUT_S) -> SparqlResult:
    """Execute a (sanitized) SPARQL SELECT/ASK query and return bindings."""
    endpoint = discover_endpoint()

    def _fetch():
        return requests.get(
            endpoint,
            params={"query": query, "format": "json"},
            headers={
                "Accept": "application/sparql-results+json",
                "User-Agent": config.USER_AGENT,
            },
            timeout=timeout,
        )

    # requests' timeout is between reads, not a deadline for the whole call: an
    # endpoint that trickles bytes holds the connection open indefinitely. One
    # question in an evaluation run took 7003 seconds against this 60s setting.
    # A hard wall-clock deadline is the only thing that actually bounds it. The
    # worker thread may linger, but the caller is freed.
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_fetch)
        try:
            resp = future.result(timeout=timeout + 5)
        except FuturesTimeout:
            raise SparqlError(
                f"SPARQL endpoint exceeded the {timeout}s deadline for this query"
            ) from None
        except requests.RequestException as exc:
            raise SparqlError(f"Network error talking to SPARQL endpoint: {exc}") from exc

    if resp.status_code != 200:
        snippet = resp.text[:500]
        raise SparqlError(f"Endpoint returned HTTP {resp.status_code}: {snippet}")

    data = resp.json()

    if "boolean" in data:  # ASK query
        return SparqlResult(columns=["boolean"], rows=[{"boolean": str(data["boolean"])}])

    columns = data.get("head", {}).get("vars", [])
    rows: list[dict[str, str]] = []
    for binding in data.get("results", {}).get("bindings", []):
        row = {}
        for var, cell in binding.items():
            value = cell.get("value", "")
            # Shorten entity URIs to readable QIDs/PIDs.
            value = value.replace(config.ENTITY_PREFIX, "").replace(
                config.PROP_DIRECT_PREFIX, ""
            )
            row[var] = value
        rows.append(row)
    return SparqlResult(columns=columns, rows=rows)
