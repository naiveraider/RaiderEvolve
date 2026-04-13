# Evolve System

LLM-guided evolutionary optimization (OpenEvolve-style loop) with Pacman and 3×3 matrix tasks, FastAPI backend, and Next.js UI.

---

## Table of Contents

- [Local Development](#local-development)
- [Docker (Production)](#docker-production)
- [Environment Variables](#environment-variables)
- [API Endpoints](#api-endpoints)
- [Tests](#tests)
- [Optional Extensions](#optional-extensions)

---

## Local Development

### 1. Backend (FastAPI + uvicorn)

Run all commands from the **repository root**:

```bash
# Create virtual environment (first time only)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API key (first time only)
cp .env.example .env             # then edit .env and fill in your key

# Start the backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

> uvicorn must be started from the project root so that `.env` is discovered automatically.

Verify the backend is running:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","service":"evolve"}
```

---

### 2. Frontend (Next.js)

Open a second terminal:

```bash
cd web
npm install        # first time only
npm run dev        # hot-reload dev server
```

Open [http://localhost:3000](http://localhost:3000).

In dev mode the frontend proxies `/api/backend/*` to the backend with an extended timeout, which is long enough to handle SSE streaming runs.

If you still see **`ECONNRESET` / `socket hang up`** errors, add the following to `web/.env.local` so the browser calls the backend directly (CORS is already open):

```bash
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

Then restart `npm run dev`.

> The `util._extend` deprecation warning printed by Node comes from a third-party dependency and is harmless. To suppress it: `NODE_OPTIONS=--no-deprecation npm run dev`

---

## Docker (Production)

### Prerequisites

- Docker and Docker Compose v2 (`docker compose` command)
- A `.env` file in the project root with a valid API key

### Start Everything (recommended)

```bash
# Build images and start all services in the background
docker compose up --build -d
```

| Service | URL |
|---------|-----|
| Frontend | [http://localhost:3000](http://localhost:3000) |
| Backend  | [http://localhost:8000](http://localhost:8000) |

View logs:

```bash
docker compose logs -f            # all services
docker compose logs -f backend    # backend only
docker compose logs -f web        # frontend only
```

Stop services:

```bash
docker compose down
```

---

### Build / Run Services Individually

**Backend only:**

```bash
docker build -t raiderevolve-backend .
docker run --rm -p 8000:8000 \
  --env-file .env \
  raiderevolve-backend
```

**Frontend only:**

```bash
docker build -t raiderevolve-web ./web \
  --build-arg BACKEND_URL=http://host.docker.internal:8000
docker run --rm -p 3000:3000 \
  -e BACKEND_URL=http://host.docker.internal:8000 \
  raiderevolve-web
```

> `host.docker.internal` lets the container reach a backend process running on the host machine. Supported on macOS and Windows Docker Desktop. On Linux add `--add-host=host.docker.internal:host-gateway` to the `docker run` command.

---

## Environment Variables

Create a `.env` file in the project root (use `.env.example` as a template):

```dotenv
# LLM API key — either name works
OPENAI_API_KEY=sk-...
# LLM_API_KEY=sk-...

# Optional: point to an OpenAI-compatible third-party endpoint
# OPENAI_BASE_URL=https://api.x.ai/v1              # xAI Grok
# OPENAI_BASE_URL=https://api.groq.com/openai/v1   # Groq

# Model to use (default: gpt-4o-mini)
# LLM_MODEL=grok-3-mini
# LLM_MODEL=llama3-8b-8192
```

If the key is missing or set to `YOUR_API_KEY`, the backend falls back to a **deterministic mock** and makes no real LLM calls.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Health check |
| `POST` | `/evolve/stream` | **SSE streaming run** — recommended; streams per-generation progress in real time |
| `POST` | `/evolve/sync` | Synchronous run — waits for all strategies to finish, then returns |
| `POST` | `/evolve` | Async job — returns a `job_id` immediately |
| `GET`  | `/evolve/{job_id}` | Poll async job status |
| `POST` | `/analytics/best-up-to` | Best fitness achieved up to a given generation |
| `POST` | `/export/fitness-csv` | Download fitness curves as CSV |

---

## Tests

```bash
# Activate the virtual environment first, then run from the project root
pytest tests -q
```

---

## Optional Extensions

- **Vector memory (Qdrant / FAISS):** `MemoryStore` currently runs in-process with code-hash deduplication. You can plug in Qdrant via the `settings.qdrant_url` hook point to enable embedding-based similarity search across the candidate history.
