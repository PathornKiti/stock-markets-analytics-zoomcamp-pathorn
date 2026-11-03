# Pipeline image: runs scripts/run_pipeline.py once and exits.
# Build context is this folder (set-quant-alpha/):
#   docker build -t set-quant-alpha .
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/models:/app/models" -v "$PWD/reports:/app/reports" \
#       set-quant-alpha --mode live --steps ingest transform signals report
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
RUN mkdir -p data/raw data/processed data/external models reports

ENTRYPOINT ["python", "scripts/run_pipeline.py"]
CMD ["--mode", "file"]
