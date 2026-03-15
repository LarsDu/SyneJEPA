FROM python:3.14-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./
COPY README.md ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy source and configs
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

ENTRYPOINT ["uv", "run", "python", "scripts/train.py"]
