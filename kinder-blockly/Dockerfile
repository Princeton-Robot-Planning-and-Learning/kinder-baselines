# syntax=docker/dockerfile:1.6

# --- Stage 1: build the Svelte frontend ---
FROM node:20-slim AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
# Vite outputs to ../src/kinder_blockly/static (see vite.config.js).
#
# SENTRY_DSN_FRONTEND is passed via fly.toml [build.args] (or
# `fly deploy --build-arg SENTRY_DSN_FRONTEND=...`). It is intentionally a
# build arg rather than a runtime env var because Vite inlines env vars at
# build time. Frontend Sentry DSNs are public — they ride along in the JS
# bundle every visitor downloads — so committing the value is fine.
ARG SENTRY_DSN_FRONTEND=""
ARG SENTRY_RELEASE=""
ENV VITE_SENTRY_DSN=${SENTRY_DSN_FRONTEND}
ENV VITE_SENTRY_RELEASE=${SENTRY_RELEASE}
RUN mkdir -p /build/src/kinder_blockly/static \
    && npm run build

# --- Stage 2: python runtime ---
FROM python:3.10-slim AS runtime

# System libraries pybullet links against. We do not need a display server;
# pybullet falls back to its built-in CPU renderer for getCameraImage when
# no GPU/X11 is available.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglu1-mesa \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libsm6 \
        libice6 \
        libxcb1 \
        libxkbcommon0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/

# Copy the pre-built frontend bundle into the package's static dir BEFORE
# pip install, so setuptools picks it up via the static/**/* package_data
# rule in pyproject.toml. Without this, pip installs an empty static dir
# into site-packages and Flask returns 404 for index.html at runtime.
COPY --from=frontend /build/src/kinder_blockly/static /app/src/kinder_blockly/static

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir gunicorn \
    && pip install --no-cache-dir .

COPY gunicorn_conf.py ./

ENV PYTHONUNBUFFERED=1 \
    PORT=8080
EXPOSE 8080

CMD ["gunicorn", "-c", "gunicorn_conf.py", "kinder_blockly.server:app"]
