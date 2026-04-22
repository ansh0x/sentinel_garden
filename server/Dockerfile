# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Sentinel Garden v2 — Multi-stage Docker build
# Supports both OpenEnv-base and standalone FastAPI modes

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml config.yaml ./
COPY server/ ./server/
COPY models.py task_registry.py client.py grader.py train.py inference.py scoring/ session_store/ logging/ ./
COPY data/ ./data/
COPY prompts/ ./prompts/
COPY __init__.py ./

# Create virtual environment and install dependencies
RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python -e ".[all]" || \
    uv pip install --python /app/.venv/bin/python -e "."

# Final runtime stage
FROM ${BASE_IMAGE}

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY --from=builder /app /app

# Environment configuration
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV SESSIONS_PATH=/app/data/sessions.json
ENV SENTINEL_CONFIG=/app/config.yaml

# Scoring mode: prelabeled | judge | hybrid
ENV SENTINEL_SCORING_MODE=prelabeled

# Deep health check

# Run the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
