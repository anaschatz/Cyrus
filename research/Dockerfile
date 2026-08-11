# syntax=docker/dockerfile:1.7
ARG JARVIS_PYTHON_IMAGE=python:3.11.9-slim-bookworm
FROM ${JARVIS_PYTHON_IMAGE} AS builder

ARG UV_VERSION=0.9.18
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1
RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"
WORKDIR /opt/research
COPY research/pyproject.toml research/uv.lock ./
COPY research/src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM ${JARVIS_PYTHON_IMAGE} AS runtime
ENV PATH=/opt/research/.venv/bin:$PATH \
    PYTHONPATH=/opt/orchestrator/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN groupadd --gid 65532 jarvis && useradd --uid 65532 --gid 65532 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin jarvis
WORKDIR /opt/research
COPY --from=builder --chown=65532:65532 /opt/research/.venv ./.venv
COPY --chown=65532:65532 orchestrator/src/jarvis_orchestrator /opt/orchestrator/src/jarvis_orchestrator
USER 65532:65532
EXPOSE 8770
HEALTHCHECK --interval=10s --timeout=2s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8770/healthz', timeout=1); raise SystemExit(0 if r.status == 200 else 1)"]
ENTRYPOINT ["jarvis-research-api"]
