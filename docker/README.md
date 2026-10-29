# Docker

The MaRDI KG QA Streamlit app is containerised here and published to the GitHub
Container Registry (ghcr.io) by the
[`docker-publish.yml`](../.github/workflows/docker-publish.yml) workflow on every
push to `main`/`master`.

The ZIB copy of the repository builds its own image through
[`.gitlab-ci.yml`](../.gitlab-ci.yml) and publishes it to that project's
container registry:

```
registry.git.zib.de/tignatenko/mardi_qa:latest     # default branch
registry.git.zib.de/tignatenko/mardi_qa:<sha>      # immutable per-commit tag
```

Deploy from the commit tag rather than `:latest` — `:latest` moves under you.

## Image

```
ghcr.io/<owner>/<repo>:latest          # default branch
ghcr.io/<owner>/<repo>:<branch>        # per-branch tag
ghcr.io/<owner>/<repo>:<branch>-<sha>  # immutable per-commit tag
```

`<owner>/<repo>` resolves automatically from `github.repository` (lowercased).

## Runtime environment variables

| Variable            | Purpose                                      | Default                        |
| ------------------- | -------------------------------------------- | ------------------------------ |
| `LLM_PROVIDER`      | `ollama` or `anthropic`                      | `ollama`                       |
| `OLLAMA_API_KEY`    | ZIB Ollama key (or paste in the sidebar)     | _(empty)_                      |
| `OLLAMA_HOST`       | Ollama endpoint                              | `https://ollama.zib.de/ollama` |
| `OLLAMA_MODEL`      | Model id — see [MODEL_CHOICE.md](../MODEL_CHOICE.md) | `qwen3-coder-next:latest` |
| `ANTHROPIC_API_KEY` | Claude key, when `LLM_PROVIDER=anthropic`    | _(empty)_                      |
| `ANTHROPIC_MODEL`   | Claude model id                              | `claude-sonnet-4-5`            |
| `ROUTER_ENABLED`    | `1` restores routing; off means every question goes through the graph, which is the path the evaluation measures | `0` |
| `APP_PASSWORD`      | Shared password gating the UI                | `zibmath2026`                  |

> `OLLAMA_API_KEY` is the one variable with no usable default: the ZIB server is
> auth-gated, so without it every question fails with a 401.

> `APP_PASSWORD` has a built-in default (`zibmath2026`), so the app works with no
> configuration. Set the env var only if you want a different password.

> The container loads the model on startup, which takes two to four minutes while
> the ZIB server pages it in. Until that finishes the first page view blocks;
> this is expected and happens once per container.

## Run the published image

```bash
docker run --rm -p 8501:8501 \
  -e OLLAMA_API_KEY=... \
  ghcr.io/<owner>/<repo>:latest
# add  -e APP_PASSWORD=your-password  only if you want to override the default
```

To run against Claude instead:

```bash
docker run --rm -p 8501:8501 \
  -e LLM_PROVIDER=anthropic -e ANTHROPIC_API_KEY=sk-ant-... \
  ghcr.io/<owner>/<repo>:latest
```

Open http://localhost:8501.

## Build & run locally

```bash
# From the repo root
docker build -f docker/Dockerfile -t mardi-qa:local .
docker run --rm -p 8501:8501 -e ANTHROPIC_API_KEY=sk-ant-... mardi-qa:local

# …or with compose
ANTHROPIC_API_KEY=sk-ant-... docker compose -f docker/docker-compose.yml up --build
```
