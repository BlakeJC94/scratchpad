ARG PYTHON_VERSION=3.12.10
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

ARG UV_VERSION=0.6.17

ENV UV_VERSION=${UV_VERSION} \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy \
    PIP_DISABLE_PIP_VERSION_CHECK=on

RUN pip3 install --user "uv==${UV_VERSION}"

# Ensure PATH includes UV bin directory
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

RUN uv tool install mlflow --with psycopg2-binary

EXPOSE 5000
ENV HOST="0.0.0.0" \
    PORT=5000

CMD ["sh", "-c", "mlflow server --host $HOST --port $PORT"]
