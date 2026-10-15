# MaRDI KG Question Answering

The pipeline's LLM is `qwen3-coder-next` (Ollama); [MODEL_CHOICE.md](MODEL_CHOICE.md)
records how it was picked and what the comparison does and does not show.

## Quick start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```

## Architecture

Every question first passes a **routing gateway** (`mardi_qa/gateway.py`),
which classifies it with a cheap LLM (heuristic fallback if that fails) and
dispatches to one of three routes:

| Route | When | What runs |
|---|---|---|
| `sparql` | facts about named entities, aggregates | full KG pipeline: entity linking → SPARQL generation → execution → grounded answer (`pipeline.py`) |
| `rag` | topic discovery, no package named ("packages for clustering") | BM25 retrieval over the cached corpus of all ~20k R-package descriptions (`rag.py`) |
| `direct` | the KG cannot answer: definitions, opinions, future, absent data | one cheap-LLM call with honesty guardrails, no retrieval |

The RAG corpus is cached at `data/rag_corpus.json` (auto-built on first use;
rebuild with `python -m mardi_qa.rag --build`). Routing is controlled by
`ROUTER_ENABLED`, `ROUTER_ANTHROPIC_MODEL` / `ROUTER_OLLAMA_MODEL` env vars —
see `mardi_qa/config.py`.

## Evaluation

- `eval/` — deterministic 25-question regression set, run on every meaningful
  change: `python eval/run_regression.py` (see `eval/README.md`).
- `evaluate.py` — multi-backend comparison with an LLM judge and PDF report.
