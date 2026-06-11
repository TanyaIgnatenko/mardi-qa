"""Central configuration for the MaRDI KG QA system."""

import os

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

# --- LLM ----------------------------------------------------------------------
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# --- Misc ---------------------------------------------------------------------
SPARQL_TIMEOUT_S = 60
DEFAULT_RESULT_LIMIT = 50          # injected if the generated query has no LIMIT
MAX_PROPERTIES_IN_PROMPT = 60      # top-k schema properties shown to the LLM
USER_AGENT = "MaRDI-QA-Streamlit/0.1 (research prototype)"
