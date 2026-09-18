# Minimal image for the standalone `stoptimes record` recorder.
#
# No Django, no Celery, no Redis, nothing agency-specific: every runtime
# choice (which source, where the archive lives, credentials) is passed at
# `docker run` time as CLI args / environment variables, never baked in here.
#
# Build from the repo root (this Dockerfile needs the uv workspace lockfile
# and the `core`/`cli`/`record` packages it resolves against):
#
#   docker build -f packages/record/Dockerfile -t stoptimes-record .
#
# Run (local filesystem archive):
#
#   docker run --rm \
#     -v "$PWD/examples/agencies:/sources:ro" \
#     -v recorder-data:/data \
#     -e STOPTIMES_SOURCES=/sources \
#     stoptimes-record --source mbta --out /data/archive --spool-dir /data/spool
#
# Run against S3/MinIO instead of a local archive: point the source's
# `archive.uri` at an `s3://...` URI and supply AWS_ENDPOINT_URL /
# AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION as `docker run -e`
# flags or an env file — never in the image or the source YAML.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

WORKDIR /app

# Only the workspace files and the three packages `stoptimes record` needs at
# runtime (cli -> record -> core); the other workspace members (train, serve,
# val, models) are never copied into this image.
COPY pyproject.toml uv.lock ./
COPY packages/core packages/core
COPY packages/record packages/record
COPY packages/cli packages/cli

RUN uv sync --frozen --no-dev --package stoptimes --extra record

# Non-root: the recorder never needs privileges beyond its own data volume.
RUN useradd --create-home --uid 10001 recorder \
    && mkdir -p /data/spool /data/archive /data/status \
    && chown -R recorder:recorder /app /data
USER recorder

ENV PATH="/app/.venv/bin:${PATH}"
VOLUME ["/data"]

ENTRYPOINT ["stoptimes", "record", "start"]
# No default --source: the image is agency-agnostic on purpose. A bare
# `docker run stoptimes-record` prints the CLI's own usage/help instead of
# silently doing nothing or defaulting to any particular agency.
CMD ["--help"]
