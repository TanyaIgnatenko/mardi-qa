"""Streamlit UI for natural-language QA over the MaRDI knowledge graph.

Run with:  streamlit run app.py
Requires:  ANTHROPIC_API_KEY in the environment (or paste it in the sidebar).
"""

import streamlit as st

from mardi_qa import config
from mardi_qa.entity_linking import fetch_property_catalog
from mardi_qa.llm import LLM
from mardi_qa.gateway import route_and_answer

st.set_page_config(page_title="MaRDI KG Question Answering", page_icon="🔎", layout="wide")


# --- Password gate ----------------------------------------------------------------
def require_password() -> None:
    """Block the UI until the user enters the shared password."""
    if st.session_state.get("authenticated"):
        return

    st.title("🔒 MaRDI KG QA")
    entered = st.text_input("Password", type="password")
    if entered:
        if entered == config.APP_PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()


require_password()

EXAMPLE_QUESTIONS = [
    "What is the purpose of the CRAN package ggplot2?",
    "Which CRAN packages are related to linear regression?",
    "Which packages does dplyr depend on?",
    "Who maintains the CRAN package data.table?",
    "What is the difference between the CRAN packages data.table and dplyr?",
]


# --- Cached resources -----------------------------------------------------------
@st.cache_data(ttl=24 * 3600, show_spinner="Loading MaRDI property catalog…")
def load_property_catalog() -> dict[str, str]:
    return fetch_property_catalog()


@st.cache_resource(show_spinner="Loading the local model (first start only)…")
def get_llm(api_key: str, model: str) -> LLM:
    # Temperature is left at the facade's default of 0, which is what every
    # evaluation run used: the app must not be a differently-tuned system from
    # the one the numbers in eval/README.md describe.
    llm = LLM(api_key=api_key, model=model)

    # Load the model here rather than on the first question. The ZIB server
    # evicts idle models and a reload took 135–218 seconds in our runs, longer
    # than OLLAMA_TIMEOUT_S: without this, the first question after an idle
    # period dies on a timeout instead of answering. Retried, because one
    # attempt can expire while the load is still running. Hosted backends load
    # nothing, so they skip it.
    if llm.provider == "ollama":
        for _ in range(3):
            try:
                llm.complete("Reply with the single word OK.", "Say OK", max_tokens=8)
                break
            except Exception:
                continue

    return llm


# --- Sidebar ----------------------------------------------------------------------
with st.sidebar:
    st.title("🔎 MaRDI KG QA")
    st.caption(
        "Ask natural-language questions about the [MaRDI knowledge graph]"
        "(https://portal.mardi4nfdi.de). Questions are translated to SPARQL, "
        "executed against the live endpoint, and answered by an LLM grounded "
        "in the retrieved graph nodes."
    )
    # Both fields follow the configured backend. They used to be hardcoded to
    # Anthropic, so switching the default provider to Ollama would have handed a
    # Claude model name to the ZIB server and asked for an API key the run does
    # not need.
    if config.LLM_PROVIDER == "ollama":
        api_key = st.text_input(
            "Ollama API key",
            value=config.OLLAMA_API_KEY,
            type="password",
            help="Or set OLLAMA_API_KEY in your environment.",
        )
        model = st.text_input("Model", value=config.OLLAMA_MODEL)
    else:
        api_key = st.text_input(
            "Anthropic API key",
            value=config.ANTHROPIC_API_KEY,
            type="password",
            help="Or set ANTHROPIC_API_KEY in your environment.",
        )
        model = st.text_input("Model", value=config.ANTHROPIC_MODEL)

    st.divider()
    st.markdown("**Endpoint**")
    try:
        from mardi_qa.sparql_client import discover_endpoint
        st.code(discover_endpoint(), language=None)
        catalog = load_property_catalog()
        st.success(f"Connected — {len(catalog)} properties in catalog.")
    except Exception as exc:
        catalog = {}
        st.error(f"Could not connect: {exc}")

    st.divider()
    st.markdown("**Example questions**")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state["pending_question"] = q


# --- Chat state -------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, trace)

st.title("Ask the MaRDI Knowledge Graph")

# Render history
for question, trace in st.session_state.history:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(trace.answer)
        _render_trace = st.session_state.get("show_trace", True)
        if _render_trace:
            with st.expander("🔬 How this answer was produced"):
                if trace.route:
                    _route_icons = {"sparql": "🕸️", "rag": "📚", "direct": "💬"}
                    st.markdown(
                        f"**Route:** {_route_icons.get(trace.route, '')} `{trace.route}`"
                        + (f" — {trace.route_reason}" if trace.route_reason else "")
                        + (f" *(via {trace.route_source})*" if trace.route_source else "")
                    )
                if trace.linked_entities:
                    st.markdown("**Linked entities**")
                    for e in trace.linked_entities:
                        st.markdown(
                            f"- [{e.qid}]({e.portal_url()}) — **{e.label}** "
                            f"({e.description or 'no description'})"
                        )
                for i, q in enumerate(trace.queries):
                    st.markdown(f"**SPARQL query {i + 1}**")
                    st.code(q, language="sparql")
                    if i < len(trace.results):
                        st.markdown(f"**Results ({len(trace.results[i].rows)} rows)**")
                        if not trace.results[i].is_empty:
                            st.dataframe(trace.results[i].rows, use_container_width=True)
                        else:
                            st.caption("No rows returned.")
                for err in trace.query_errors:
                    st.warning(err)
                if trace.used_fallback:
                    st.info(
                        "SPARQL retrieval was insufficient — the answer was grounded "
                        "in the full claim sets of the linked entities instead."
                    )
                    st.text(trace.fallback_context[:4000])

# --- Input ------------------------------------------------------------------------
question = st.chat_input("e.g. Who maintains the CRAN package data.table?")
if not question and "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:
    if not api_key:
        st.error(f"Please provide a {config.LLM_PROVIDER} API key in the sidebar.")
        st.stop()
    if not catalog:
        st.error("Property catalog unavailable — cannot ground SPARQL generation.")
        st.stop()

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        status = st.status("Working…", expanded=True)

        def progress(msg: str):
            status.write(msg)

        try:
            llm = get_llm(api_key, model)
            trace = route_and_answer(question, llm, catalog, progress=progress)
            status.update(label="Done", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Failed", state="error")
            st.error(f"Pipeline error: {exc}")
            st.stop()

        st.session_state.history.append((question, trace))
        st.rerun()
