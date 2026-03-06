# ======================
# Evolution Settings
# ======================

POP_SIZE = 4
TOP_K = 2
GENERATIONS = 5

# Selection: "best" (single best) or "top_k" (keep top k candidates)
SELECTION_MODE = "top_k"

# Candidate generation: "perturb" | "templates" | "swap" | "human" | "llm" | "hybrid"
# - perturb: randomly perturb parameters (e.g. pivot selection)
# - templates: replace from predefined templates
# - swap: simple mutation (e.g. swap two lines)
# - human: human-in-the-loop (no-op unless hook provided)
# - llm: prompt-based LLM improvement (mandatory minimum)
# - hybrid: mix of strategies (see HYBRID_STRATEGIES)
CANDIDATE_GENERATION = "hybrid"

# For hybrid mode: strategy -> probability (should sum to 1.0)
HYBRID_STRATEGIES = {
    "perturb": 0.25,
    "templates": 0.25,
    "llm": 0.50,
}

# Pacman fitness: fitness = w1*score + w2*survival_time - w3*cost(steps), sum(weights)=1
PACMAN_FITNESS_WEIGHTS = (0.5, 0.3, 0.2)  # (w1=score, w2=survival_time, w3=cost)

# ======================
# LLM Settings
# ======================

USE_LLM = True
LLM_MUTATION_RATIO = 0.3

LLM_MODEL = "gpt-4o-mini"  # example
LLM_API_KEY = "YOUR_API_KEY"  # or read from env