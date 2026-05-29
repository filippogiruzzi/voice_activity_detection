# syntax=docker/dockerfile:1
FROM vad:latest

WORKDIR /app

CMD ["sh", "-c", "uv run ruff check . && uv run ruff format --check . && uv run pytest"]
