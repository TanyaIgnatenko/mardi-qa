"""Streamlit UI for natural-language QA over the MaRDI knowledge graph.

Run with:  streamlit run app.py
Requires:  ANTHROPIC_API_KEY in the environment (or paste it in the sidebar).
"""

import streamlit as st

from mardi_qa import config
from mardi_qa.entity_linking import fetch_property_catalog
from mardi_qa.llm import LLM
from mardi_qa.pipeline import answer_question

st.set_page_config(page_title="MaRDI KG Question Answering", page_icon="🔎", layout="wide")

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


@st.cache_resource
def get_llm(api_key: str, model: str) -> LLM:
    return LLM(api_key=api_key, model=model)


# --- Sidebar ----------------------------------------------------------------------
with st.sidebar:
    st.title("🔎 MaRDI KG QA")
    st.caption(
        "Ask natural-language questions about the [MaRDI knowledge graph]"
        "(https://portal.mardi4nfdi.de). Questions are translated to SPARQL, "
        "executed against the live endpoint, and answered by an LLM grounded "
        "in the retrieved graph nodes."
    )
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
        st.error("Please provide an Anthropic API key in the sidebar.")
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
            trace = answer_question(question, llm, catalog, progress=progress)
            status.update(label="Done", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Failed", state="error")
            st.error(f"Pipeline error: {exc}")
            st.stop()

        st.session_state.history.append((question, trace))
        st.rerun()
