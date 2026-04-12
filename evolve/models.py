from __future__ import annotations

from enum import Enum
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    PACMAN = "pacman"
    MATRIX = "matrix"


class SelectionMode(str, Enum):
    TOP_K = "top_k"
    ELITE = "elite"
    DIVERSITY = "diversity"


class FitnessPreset(str, Enum):
    PACMAN = "pacman"
    MATRIX = "matrix"
    CUSTOM = "custom"


class EvolutionStrategy(str, Enum):
    SINGLE_LLM = "single_llm"
    RANDOM_ONLY = "random_only"
    FULL = "full"


class CustomFitnessWeights(BaseModel):
    w1: float = Field(0.5, ge=0.0, le=1.0)
    w2: float = Field(0.3, ge=0.0, le=1.0)
    w3: float = Field(0.2, ge=0.0, le=1.0)


class EvolutionRequest(BaseModel):
    task: TaskType
    source_code: str = ""
    generations: int = Field(8, ge=1, le=200)
    population_size: int = Field(6, ge=2, le=32)
    top_k: int = Field(3, ge=1, le=16)
    selection_mode: SelectionMode = SelectionMode.DIVERSITY
    fitness_preset: FitnessPreset = FitnessPreset.PACMAN
    custom_weights: Optional[CustomFitnessWeights] = None
    matrix_alpha: float = Field(0.01, ge=0.0)
    matrix_beta: float = Field(0.005, ge=0.0)
    strategies: Optional[List[EvolutionStrategy]] = None
    include_pseudocode_log: bool = True
    seed: Optional[int] = None


class CandidateRecord(BaseModel):
    id: str
    generation: int
    code: str
    fitness: float
    parents: List[str] = Field(default_factory=list)
    strategy_tag: str = ""
    mutation_notes: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)


class GenerationLog(BaseModel):
    generation: int
    selection_summary: str
    mutation_explanations: List[str]
    best_id: str
    avg_fitness: float
    best_fitness: float


class StrategyRunResult(BaseModel):
    strategy: EvolutionStrategy
    best_code: str
    fitness_curve: List[float]
    avg_fitness_per_gen: List[float]
    best_per_generation: List[float]
    history: List[GenerationLog]
    memory_records: List[CandidateRecord]
    final_best_fitness: float


class EvolutionResponse(BaseModel):
    task: TaskType
    runs: List[StrategyRunResult]
    pseudocode_outline: Optional[str] = None
    algorithm_explanation: Optional[str] = None


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "done", "error"]
    progress: float = 0.0
    message: str = ""
    result: Optional[EvolutionResponse] = None
    error: Optional[str] = None
