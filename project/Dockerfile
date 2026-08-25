# Build context is the REPO ROOT, not project/:
#   docker build -t algotrade -f project/Dockerfile .
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Dependency layer first so code edits don't invalidate the install cache.
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY project/src/ ./project/src/
RUN uv sync --locked --no-group dev

COPY project/ ./project/

CMD ["uv", "run", "--no-sync", "project/scripts/run_pipeline.py", "--mode", "live"]
