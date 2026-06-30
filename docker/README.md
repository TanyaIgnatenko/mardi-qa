# Docker

The MaRDI KG QA Streamlit app is containerised here and published to the GitHub
Container Registry (ghcr.io) by the
[`docker-publish.yml`](../.github/workflows/docker-publish.yml) workflow on every
push to `main`/`master`.

## Image

```
ghcr.io/<owner>/<repo>:latest          # default branch
ghcr.io/<owner>/<repo>:<branch>        # per-branch tag
ghcr.io/<owner>/<repo>:<branch>-<sha>  # immutable per-commit tag
```

`<owner>/<repo>` resolves automatically from `github.repository` (lowercased).

## Runtime environment variables

| Variable            | Purpose                                  | Default              |
| ------------------- | ---------------------------------------- | -------------------- |
| `ANTHROPIC_API_KEY` | Claude API key (or paste in the sidebar) | _(empty)_            |
| `ANTHROPIC_MODEL`   | Claude model id                          | `claude-sonnet-4-5`  |
| `APP_PASSWORD`      | Shared password gating the UI            | `zibmath2026`        |

> `APP_PASSWORD` has a built-in default (`zibmath2026`), so the app works with no
> configuration. Set the env var only if you want a different password.

## Run the published image

```bash
docker run --rm -p 8501:8501 \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  ghcr.io/<owner>/<repo>:latest
# add  -e APP_PASSWORD=your-password  only if you want to override the default
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
