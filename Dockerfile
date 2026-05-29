# syntax=docker/dockerfile:1
ARG BASE_IMAGE=python:3.11-slim

FROM ${BASE_IMAGE}

LABEL author="filippo.giruzzi@gmail.com"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml .python-version uv.lock* ./

# Install dependencies (without the project itself)
RUN uv sync --frozen --no-install-project

# Copy project source
COPY . .

# Install the project in editable mode
RUN uv sync --frozen

# Verify import
RUN uv run python -c "import vad"
