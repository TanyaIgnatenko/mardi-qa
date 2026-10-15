# Model choice

**Chosen: `qwen3-coder-next:latest`** (Ollama, ZIB deployment), to replace
`claude-sonnet-4-5` for the RAG pipeline.

## How it was picked

Six Ollama candidates were screened in two steps.

**Step 1 — simple / can it produce what the pipeline needs?** (`eval/gate_models.py`)
The pipeline asks the model for two things: a list of entities as JSON, and a
SPARQL query. If either comes back broken, nothing downstream can run. Three
models of six passed; the other three wrote queries the endpoint refused.

**Step 2 — easy set** (`eval/run_phase1.py --set easy`): five single-hop
questions with gold answers computed from the graph. Each model run twice at
temperature 0, identical prompts and sampling settings.

| model | correct query | fact asserted | fabricated links | per question | reproducible |
|---|---|---|---|---|---|
| **qwen3-coder-next** | **5/5** | **4/5** | **1** | **10 s** | **yes** |
| claude-sonnet-4-5 | 5/5 | 4/5 | 1 | 13 s | yes |
| deepseek-r1:14b | 5/5 → 4/5 | 4/5 → 3/5 | 2 → 3 | 37 s | no |
| qwen3.5:35b | 3/5 → 4/5 | 2/5 → 3/5 | 2 | 18 s | no |

## Why this one

- **Matches Claude exactly** on every column of the easy set.
- **Reproducible**: both runs identical question by question. The two other
  local candidates gave different answers to the same question across runs,
  which also means a user who asks twice gets two answers.
- **Faster**: ~10 s per question against ~13 s, with no API cost.
