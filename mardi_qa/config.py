"""Central configuration for the MaRDI KG QA system."""

import os
from pathlib import Path

# Loaded here, the one module every entry point imports, because only the eval
# harness used to read .env: with Anthropic the README told you to export the
# key by hand, but the default backend is now the auth-gated ZIB server, so
# `streamlit run app.py` would fail on a 401 without this. Existing environment
# variables win, so a deployment that sets them directly is unaffected.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

# --- MaRDI Wikibase endpoints -------------------------------------------------
# Candidate SPARQL endpoint paths (Wikibase deployments vary). On startup the
# app probes these in order with a trivial query and uses the first that works.
# Set MARDI_SPARQL_ENDPOINT to skip autodiscovery.
SPARQL_ENDPOINT_CANDIDATES = [
    "https://query.portal.mardi4nfdi.de/sparql",
    "https://query.portal.mardi4nfdi.de/bigdata/namespace/wdq/sparql",
    "https://query.portal.mardi4nfdi.de/proxy/wdqs/bigdata/namespace/wdq/sparql",
    "https://portal.mardi4nfdi.de/query/sparql",
]
SPARQL_ENDPOINT = os.getenv("MARDI_SPARQL_ENDPOINT", "")

# MediaWiki action API of the portal (entity search / labels).
MEDIAWIKI_API = os.getenv(
    "MARDI_MEDIAWIKI_API",
    "https://portal.mardi4nfdi.de/w/api.php",
)

# Concept URI bases of the MaRDI Wikibase (NOT wikidata.org!).
ENTITY_PREFIX = "https://portal.mardi4nfdi.de/entity/"
PROP_DIRECT_PREFIX = "https://portal.mardi4nfdi.de/prop/direct/"

SPARQL_PREFIXES = f"""\
PREFIX wd: <{ENTITY_PREFIX}>
PREFIX wdt: <{PROP_DIRECT_PREFIX}>
PREFIX p: <https://portal.mardi4nfdi.de/prop/>
PREFIX ps: <https://portal.mardi4nfdi.de/prop/statement/>
PREFIX pq: <https://portal.mardi4nfdi.de/prop/qualifier/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX mwapi: <https://www.mediawiki.org/ontology#API/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
"""

# --- LLM backend selection ----------------------------------------------------
# "ollama" (default, ZIB local models) or "anthropic" (Claude).
# See MODEL_CHOICE.md for the comparison behind the default.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# --- Anthropic (Claude) -------------------------------------------------------
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# --- Ollama (ZIB) -------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.zib.de/ollama")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3-coder-next:latest")
# Generous because this bounds a cold model load, not just generation: loading
# qwen3-coder-next took 135–218 seconds, so the old 180 could expire mid-load.
# Warm generation on the same model runs in under 15 seconds.
OLLAMA_TIMEOUT_S = int(os.getenv("OLLAMA_TIMEOUT_S", "420"))

# --- Routing gateway ----------------------------------------------------------
# The router classifies each question with a CHEAP model before the expensive
# pipeline runs; the "direct" route answers with that same cheap model.
# Off by default so the deployed app runs the path the evaluation measures: the
# harness forces ROUTER_ENABLED=0 and sends every question through the graph, so
# with routing on, the app could answer from the package corpus or from the model
# alone and the numbers in eval/README.md would describe a different system.
# Trade-off: every question now reaches the graph pipeline, including ones the
# graph cannot answer. Set ROUTER_ENABLED=1 to put routing back.
ROUTER_ENABLED = os.getenv("ROUTER_ENABLED", "0") not in ("0", "false", "no")
ROUTER_ANTHROPIC_MODEL = os.getenv("ROUTER_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
ROUTER_OLLAMA_MODEL = os.getenv("ROUTER_OLLAMA_MODEL", "")  # empty -> same as OLLAMA_MODEL

# --- RAG route ------------------------------------------------------------------
# Disk cache of the package corpus (label + description of every R package).
RAG_CORPUS_PATH = os.getenv(
    "MARDI_RAG_CORPUS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "data", "rag_corpus.json"),
)
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "20"))

# --- Access control -----------------------------------------------------------
# Single shared password gating the UI. Defaults to a fixed value so the app
# works out of the box; override via the APP_PASSWORD env var if desired.
APP_PASSWORD = os.getenv("APP_PASSWORD", "zibmath2026")

# --- Misc ---------------------------------------------------------------------
SPARQL_TIMEOUT_S = 60
DEFAULT_RESULT_LIMIT = 50          # injected if the generated query has no LIMIT
# How many result rows are rendered into the answer prompt. Kept equal to the
# query limit on purpose: when this was lower (40 against a limit of 50) a tenth
# of every full result set never reached the model, so an answer could miss a
# fact its own query had retrieved.
CONTEXT_ROWS_IN_PROMPT = int(os.getenv("CONTEXT_ROWS_IN_PROMPT", str(DEFAULT_RESULT_LIMIT)))
MAX_PROPERTIES_IN_PROMPT = 60      # top-k schema properties shown to the LLM
USER_AGENT = "MaRDI-QA-Streamlit/0.1 (research prototype)"
