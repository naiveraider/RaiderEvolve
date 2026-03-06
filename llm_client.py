import copy
import os
import json
import re
from config import USE_LLM, LLM_MODEL, LLM_API_KEY

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


class LLMClient:
    """Prompt-based LLM improvement for candidates (mandatory minimum expectation)."""

    def __init__(self):
        self.enabled = USE_LLM
        self._client = None
        api_key = os.environ.get("OPENAI_API_KEY") or LLM_API_KEY
        if _OPENAI_AVAILABLE and api_key and api_key != "YOUR_API_KEY":
            try:
                self._client = OpenAI(api_key=api_key)
            except Exception:
                self._client = None

    def improve(self, problem_name, candidate):
        if not self.enabled:
            return candidate

        improved = copy.deepcopy(candidate)

        if self._client:
            improved = self._call_llm(problem_name, candidate, improved)
        else:
            improved = self._mock_improve(problem_name, improved)

        return improved

    def _call_llm(self, problem_name, candidate, improved):
        """Call small LLM API with prompt-based improvement."""
        prompt = self._build_prompt(problem_name, candidate)
        try:
            response = self._client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You suggest one improved configuration. Reply with a short JSON object only, no markdown."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=256,
            )
            text = (response.choices[0].message.content or "").strip()
            parsed = self._parse_improvement(problem_name, text, improved)
            if parsed:
                improved.update(parsed)
        except Exception:
            improved = self._mock_improve(problem_name, improved)
        return improved

    def _build_prompt(self, problem_name, candidate):
        if problem_name == "pacman":
            return (
                f"Current Pacman search config: algorithm={candidate.get('algorithm')}, heuristic={candidate.get('heuristic')}. "
                "Suggest one improved config. Reply JSON only, e.g. {\"algorithm\": \"astar\", \"heuristic\": \"manhattanHeuristic\"}."
            )
        if problem_name == "sorting":
            return (
                f"Current sorting config: pivot_strategy={candidate.get('pivot_strategy')}. "
                "Suggest one improved strategy. Reply JSON only, e.g. {\"pivot_strategy\": \"random\"}. Valid: first, random, middle, median_of_three."
            )
        return f"Improve this candidate: {candidate}. Reply with JSON only."

    def _parse_improvement(self, problem_name, text, fallback):
        """Extract JSON from LLM response and return only allowed keys."""
        # Strip markdown code blocks if present
        text = re.sub(r"^```\w*\s*", "", text).strip()
        text = re.sub(r"\s*```\s*$", "", text).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        if problem_name == "pacman":
            return {k: v for k, v in data.items() if k in ("algorithm", "heuristic") and isinstance(v, str)}
        if problem_name == "sorting":
            if "pivot_strategy" in data and isinstance(data["pivot_strategy"], str):
                return {"pivot_strategy": data["pivot_strategy"]}
        return data if isinstance(data, dict) else None

    def _mock_improve(self, problem_name, candidate):
        """Fallback when API is not available."""
        if problem_name == "pacman":
            candidate["heuristic"] = "manhattanHeuristic"
        elif problem_name == "sorting":
            candidate["pivot_strategy"] = "random"
        return candidate
