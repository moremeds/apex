# apex API server image — REST + WS chart/signal API (:8322) consumed by argon.
# Deploys to the arm64 macmini; built natively on ubuntu-24.04-arm in release.yml.
#
# Reads the livewire bronze lake from APEX_LIVEWIRE_ROOT (a read-only bind-mount in
# docker-compose.yml). Reaches host Postgres + xenon WS via host.docker.internal.
#
# Local smoke build (on the macmini / any arm64 Docker host):
#   docker build -f docker/api.Dockerfile -t apex-api:dev .
#   docker run --rm -p 8322:8322 \
#     -v /Volumes/DATA_LAKE/livewire/data-lake/bronze:/data/livewire:ro \
#     -e APEX_LIVEWIRE_ROOT=/data/livewire apex-api:dev
#   curl http://localhost:8322/health

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# TA-Lib C library 0.6.4 from source — the `TA-Lib` Python wheel is a Cython wrapper
# that needs libta_lib + headers at build AND runtime. Version matches the proven
# build in .github/workflows/release-holdout-gate.yml. curl stays in the final image
# for the compose healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential wget curl ca-certificates \
    && wget -q https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz \
    && tar -xzf ta-lib-0.6.4-src.tar.gz \
    && cd ta-lib-0.6.4 \
    && ./configure --prefix=/usr \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig \
    && cd .. && rm -rf ta-lib-0.6.4 ta-lib-0.6.4-src.tar.gz \
    && apt-get purge -y build-essential wget \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# uv, the only sanctioned installer (CLAUDE.md: uv exclusively).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Full app context. The editable install needs the source present (hatchling builds
# the project), and `python main.py` runs from /app with `src/` on sys.path — the
# same editable + run-from-repo layout the macmini uses today.
COPY pyproject.toml uv.lock README.md VERSION main.py ./
COPY src/ ./src/
COPY config/ ./config/

# `build-essential` is needed transiently to compile the TA-Lib Cython wrapper, then
# purged to keep the runtime image lean.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && uv pip install --system -e ".[api,observability]" \
    && apt-get purge -y build-essential && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Non-root runtime.
RUN useradd --create-home --uid 10001 apex && chown -R apex:apex /app
USER apex

EXPOSE 8322

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=30s \
    CMD curl -f http://localhost:8322/health || exit 1

CMD ["python", "main.py", "--service", "api"]
