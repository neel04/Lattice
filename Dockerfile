FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PORT=8000

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app.py ./app.py
COPY templates ./templates

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "python", "app.py"]
