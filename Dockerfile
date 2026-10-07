# ── Stage 1: build ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps needed for Prophet / statsmodels compile
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Runtime system library for Prophet
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy source
COPY src/ ./src/
COPY app/ ./app/
COPY data/ ./data/

# Create a non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose Streamlit (8501) and FastAPI (8000)
EXPOSE 8501 8000

# Default: run the Streamlit dashboard
# Override CMD to run the API instead:
#   docker run ... python src/api.py
ENV PYTHONPATH=/app/src
CMD ["streamlit", "run", "app/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
