# Django + Celery for GTFS-RT ingestion.
#
# IMPORTANT: build context is git.no_sync/ (see docker-compose.yml's
# `context: ../..`), the common parent of the two sibling repos this image
# needs. gtfs-eta was extracted from gtfs-django and is a standalone repo,
# but gtfs-rt-pipeline/sch_pipeline/models.py still imports gtfs.models
# (Base* GTFS models) from the gtfs-django repo at runtime, and
# gtfs-rt-pipeline/pyproject.toml resolves that as an editable path source
# at `../../gtfs-django`. This image mirrors that same sibling layout so the
# relative path resolves identically on host and in-container:
#   /app/gtfs-django                      -> gtfs-django repo root (gtfs/ + pyproject), read-only source
#   /app/gtfs-eta/gtfs-rt-pipeline        -> this ingest project
#   /app/gtfs-eta/feature_engineering
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
        libpq-dev \
        gdal-bin \
        libgdal-dev \
        binutils \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install --no-cache-dir uv

# MinIO client — rt_pipeline.compaction.storage.McStorage shells out to `mc`
# for the list/move/remove operations DuckDB cannot do itself. Without it the
# daily compaction task dies with FileNotFoundError.
RUN curl -fsSL https://dl.min.io/client/mc/release/linux-amd64/mc \
        -o /usr/local/bin/mc \
    && chmod +x /usr/local/bin/mc

# --- gtfs-django package (target of the editable path dep `gtfs-django @ ../../gtfs-django`) ---
WORKDIR /app/gtfs-django
COPY gtfs-django/pyproject.toml gtfs-django/README.md gtfs-django/LICENSE ./
COPY gtfs-django/gtfs ./gtfs

# --- ingest project: install deps (resolves gtfs-django editable from /app/gtfs-django) ---
WORKDIR /app/gtfs-eta/gtfs-rt-pipeline
COPY gtfs-eta/gtfs-rt-pipeline/pyproject.toml gtfs-eta/gtfs-rt-pipeline/uv.lock* ./
RUN uv pip install --system -e .

# --- source: the feature builder (imported by build_eta_sample) + the ingest app ---
# feature_engineering/dataset_builder.py:12 imports core.config (agency
# timezone/holiday defaults) — core/ must sit alongside feature_engineering
# under /app/gtfs-eta for that import to resolve.
COPY gtfs-eta/feature_engineering /app/gtfs-eta/feature_engineering
COPY gtfs-eta/core /app/gtfs-eta/core
COPY gtfs-eta/gtfs-rt-pipeline/ ./

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
