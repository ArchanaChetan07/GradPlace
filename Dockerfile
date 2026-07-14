FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    PYTHONPATH=/app

WORKDIR /app

# System deps for matplotlib
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY placement.py main.py test.py check_cuda.py ./
COPY scripts ./scripts
COPY tests ./tests
COPY assets ./assets

# Default: reproduce synthetic-toy before/after metrics + plots into /app/assets
CMD ["python", "scripts/eval_before_after.py", "--epochs", "2000", "--outdir", "assets", "--device", "cpu"]
