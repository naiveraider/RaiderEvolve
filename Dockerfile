# ── Backend: FastAPI + uvicorn ────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System deps needed by numpy / uvicorn
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY evolve/ ./evolve/
COPY main.py .

# Non-root user for security
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser /app
USER appuser

EXPOSE 8000

# OPENAI_API_KEY (and optional OPENAI_BASE_URL, LLM_MODEL) must be injected
# at runtime via --env-file or -e flags / compose env_file.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--timeout-keep-alive", "120"]
