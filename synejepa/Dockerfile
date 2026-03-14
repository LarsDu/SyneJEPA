FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

RUN pip install --no-cache-dir -e ".[dev]"

ENTRYPOINT ["python", "scripts/train.py"]
