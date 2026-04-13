# Round 2 Prototype - README

**Course:** CS-5381-D01
**Group:** No.3
**Members:** Lirong Chen, Jacob Ray, Zhuo Qian, MD A Rahman
**Project:** Evolve - LLM-Guided Evolutionary Algorithm for Pacman

## What is in this zip

```
group3_prototype.zip
|- PROTOTYPE_README.md         (this file)
|- RaiderEvolve/               (full source code, see RaiderEvolve/README.md)
|  |- main.py
|  |- evolve/                  (controller, evaluator, mutations, memory)
|  |- web/                     (Next.js frontend)
|  |- tests/
|  |- requirements.txt
|  |- Dockerfile
|  |- docker-compose.yml
|- data/                       (per-member data reports)
   |- mdarahman_data.csv       MD A Rahman
   |- mdarahman_data.docx
   |- lirong_chen_data.csv     Lirong Chen
   |- lirong_chen_data.docx
   |- jacob_ray_data.csv       Jacob Ray
   |- jacob_ray_data.docx
   |- zhuo_qian_data.csv       Zhuo Qian
   |- zhuo_qian_data.docx
```

Each of the four group members contributed one per-member data report.
Before submitting, the team lead should check that all four are in the
archive.

## Required environment

| Component        | Version                             |
|------------------|-------------------------------------|
| Operating system | macOS, Linux, or Windows 10+        |
| Python           | 3.11 or newer                       |
| Node.js          | 20 LTS or newer                     |
| Docker Engine    | 24+ (optional, for one-command run) |
| Docker Compose   | v2 (`docker compose`)               |
| RAM              | 8 GB minimum, 16 GB recommended     |
| CPU              | 4-core, no GPU needed               |

## Adopted libraries

Backend (Python), from `RaiderEvolve/requirements.txt`:

- `fastapi` for the HTTP and SSE server
- `uvicorn[standard]` as the ASGI server for FastAPI
- `pydantic` for request and response schemas
- `openai` for the LLM client (also works with xAI Grok and Groq)
- `python-dotenv` for the `.env` loader
- `httpx` as the async HTTP client
- `pytest` for tests
- `python-docx` and `matplotlib` for the per-member data reports

Frontend (Next.js + React), from `RaiderEvolve/web/package.json`:

- `next`, `react`, `react-dom`
- `recharts` for the fitness chart
- `tailwindcss` for styling

## Flow of execution

```
user (browser)
    |
    v
Next.js UI (localhost:3000)
    |  POST /evolve/stream
    v
FastAPI (localhost:8000)
    |
    v
controller.run_evolution_run()
    |
    +-- seed population (user source or DFS baseline)
    |
    +-- for each generation:
    |     1. selection.select_parents  (top_k / elite / diversity)
    |     2. mutation  (LLM + random + template)
    |     3. pacman_env.pacman_fitness in sandbox (3s timeout)
    |     4. memory_store.add  (dedup by code hash)
    |     5. emit SSE progress event
    |
    +-- return final best candidate + memory log
              |
              v
UI renders fitness / steps / runtime curves and Evolution History
```

## Commands to run the prototype

### Option 1: local (two terminals)

```bash
cd RaiderEvolve

# Backend
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # edit .env and add OPENAI_API_KEY
uvicorn main:app --host 127.0.0.1 --port 8000

# Frontend (second terminal)
cd RaiderEvolve/web
npm install
npm run dev
# open http://localhost:3000
```

### Option 2: Docker Compose (one command)

```bash
cd RaiderEvolve
cp .env.example .env                 # edit .env and add OPENAI_API_KEY
docker compose up --build -d
# open http://localhost:3000
```

### Running the test suite

```bash
cd RaiderEvolve
source .venv/bin/activate
pytest tests -q
```

### Per-member data reports

Each of the four members picked one parameter to test on the Pacman task,
ran a small sweep with `seed=42` for reproducibility, and wrote up the
result in `{member}_data.csv` and `{member}_data.docx`. The csv has the
raw per-generation metrics and the docx has the analysis with the plots
embedded.

## Issues we ran into, and how we fixed them

### Issue 1: request timeouts (ECONNRESET, "socket hang up")

Long evolution runs take more than a minute per strategy and the Next.js
dev proxy was killing the connection at its default 30 second timeout.

Fix: we extended `proxyTimeout` to 5 minutes and switched the main code
path from a blocking `POST /evolve/sync` call over to Server-Sent Events
(`POST /evolve/stream`). The SSE connection stays alive through the
whole run and streams a progress event after every generation.

### Issue 2: no UI feedback during long runs

The UI froze with no indication that anything was happening, which made
it look broken.

Fix: the SSE endpoint now emits a progress event after every generation.
The frontend listens to the stream, updates the fitness chart, and
appends rows to the Evolution History panel as they come in. We also
added an `AbortController`-backed Cancel button so the user can stop a
run cleanly.

### Issue 3: LLM rate limits on large sweeps

Running many strategies and many generations in parallel was hitting the
provider's TPM / RPM caps.

Fix: the memory store hashes each candidate's code and deduplicates, so
revisiting an already-evaluated function never calls the LLM again. We
also run per-member sweeps sequentially instead of in parallel to stay
under the rate limit.

### Issue 4: w1 and w2 in the fitness function do not actually do anything

We expose three weights (w1, w2, w3) in the UI and API but the evaluator
at `evolve/pacman_env.py` line 235 only uses w3 in the formula:
`fitness = avg_score - w3 * avg_cost * 0.01` (after normalization). So
you can set very different w1 and w2 values and get the same score.

This is documented as a known limitation. The next step is to wire w1
and w2 into the formula, for example
`w1 * avg_score + w2 * success_rate - w3 * avg_cost`, so all three
weights actually change the score.

## Suggestions for future iterations

1. Add procedurally generated mazes so the evolved agents are tested on
   maps they have not seen before, not just `mudMaze` and `largeMudMaze`.
   This would catch overfitting.
2. Swap the hosted LLM for a local open-source model (for example Llama
   3) to cut API cost and latency on long runs.
3. Extend the engine to co-evolve multiple agents in the same maze (for
   example Pacman + ghosts) so strategies are pushed to generalize
   against opponents.
4. Optional vector memory backend (Qdrant or FAISS) for cross-run
   candidate reuse. A hook point is already reserved in
   `evolve/settings.py`.
5. Finish the three-term fitness formula from Issue 4.

## Feedback on the project experience

The thing that made the OpenEvolve-style loop click for us was pairing
a full evolution run against a random-only baseline inside the same UI.
That is what turned "does the LLM actually help" from a vibes question
into a concrete, reproducible number: about +7.3% fitness improvement
over the random baseline on the Pacman weighted-mud mazes, and the gap
shows up in every parameter configuration we tested.

The hardest parts of the build were keeping the long HTTP connection
alive through multi-minute runs (fixed with SSE) and making the loop
visible enough that a grader can see what is happening instead of
watching a spinner. The live Evolution History panel was the most
impactful UX fix.

If we had another week the top priority would be finishing the fitness
function (Issue 4) and benchmarking on procedurally generated mazes so
we can measure generalization instead of convergence on fixed maps.
