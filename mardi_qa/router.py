"""Routing gateway: decide which answer path a user question takes.

Routes:
  sparql — factual questions about specific named entities or aggregates whose
           answer is stored in the knowledge graph (maintainer, authors,
           imports, license, similar-to, counts). The full KG pipeline runs.
  rag    — topic/thematic discovery ("packages related to X") with no specific
           package named. BM25 retrieval over the package corpus.
  direct — everything the KG cannot answer: general-knowledge definitions,
           greetings/meta questions, opinions, future events, data the graph
           does not hold (e.g. download statistics). A cheap LLM answers with
           explicit disclaimers instead of the expensive pipeline fabricating.

The classifier is a cheap LLM call returning strict JSON. If that call fails
or returns garbage, a keyword heuristic decides; its fallback default is
"sparql" — the safest route, since its honest failure mode is "no results"
rather than fabrication.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import prompts
from .llm import LLM

ROUTES = ("sparql", "rag", "direct")


@dataclass
class RouteDecision:
    route: str            # one of ROUTES
    reason: str           # one line, shown in the UI trace
    source: str           # "llm" or "heuristic"


# Phrases that signal topic discovery rather than a specific entity's facts.
_TOPIC_RE = re.compile(
    r"\b(related to|packages? for|deal(?:ing)? with|about|regarding|"
    r"which (?:r |cran )?packages)\b", re.I
)
# Phrases the KG can never answer.
_UNANSWERABLE_RE = re.compile(
    r"\b(better|best|recommend|should i|will (?:be|maintain)|in 20[3-9]\d|"
    r"download(?:s|ed)?|popular(?:ity)?|how do i|what is (?:a |an )?(?!the )|"
    r"hello|hi\b|thanks?)\b", re.I
)


def _heuristic(question: str) -> RouteDecision:
    if _UNANSWERABLE_RE.search(question):
        return RouteDecision("direct", "heuristic: opinion/meta/out-of-graph phrasing", "heuristic")
    if _TOPIC_RE.search(question) and not re.search(r"\b[a-z]+\d|[A-Z][a-z]+[A-Z]", question):
        return RouteDecision("rag", "heuristic: topic phrasing, no specific package named", "heuristic")
    return RouteDecision("sparql", "heuristic default: treat as a factual KG question", "heuristic")


def decide_route(question: str, router_llm: LLM | None) -> RouteDecision:
    """Classify with the cheap LLM; fall back to the keyword heuristic."""
    if router_llm is not None:
        try:
            data = router_llm.complete_json(prompts.ROUTER_SYSTEM, question)
            route = str(data.get("route", "")).strip().lower()
            if route in ROUTES:
                return RouteDecision(route, str(data.get("reason", ""))[:200], "llm")
        except Exception:
            pass
    return _heuristic(question)
