# The replayer and the twin service (`dte publish`, `dte serve`); see docker-compose.yml.
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim
# git: PAS comes from GitHub; libgomp1: LightGBM's OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# only the lock files before installing dependencies, so other edits reuse that layer
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY README.md ./
COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev
ENV VITALDB_CACHE_DIR=/app/data/vitaldb
ENTRYPOINT ["uv", "run", "--no-sync", "dte"]
