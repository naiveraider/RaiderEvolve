CS-5381 Project – Evolutionary Algorithm with LLM

## Run server
```bash
uvicorn api:app --reload
```

## API
- `POST /solve/pacman` – evolve Pacman search agent
- `POST /solve/sorting` – evolve sorting (pivot) strategy

## Config (`config.py`)

**Selection**
- `SELECTION_MODE`: `"best"` (single best) or `"top_k"` (keep top k)

**Candidate generation**
- `CANDIDATE_GENERATION`: `"perturb"` | `"templates"` | `"swap"` | `"human"` | `"llm"` | `"hybrid"`
  - **perturb**: randomly perturb parameters (e.g. pivot selection)
  - **templates**: replace from predefined templates in `problems/templates.py`
  - **swap**: simple mutation (e.g. swap two config choices)
  - **human**: human-in-the-loop (use `EvolutionEngine(..., human_hook=fn)`)
  - **llm**: prompt-based LLM improvement (mandatory minimum)
  - **hybrid**: mix of strategies via `HYBRID_STRATEGIES`

**Pacman fitness**
- `fitness = w1×score + w2×survival_time − w3×cost(steps)`, weights in `PACMAN_FITNESS_WEIGHTS`

**LLM**
- Set `OPENAI_API_KEY` or `LLM_API_KEY` in config for real API; otherwise mock improvement is used.


