"""LLM-driven Credit Calculus experiments (Qwen3.5-0.8B via vLLM)."""

from .config import LLMConfig, DEFAULT_LLM_CONFIG, PlanningCondition
from .swarm import LLMSwarmSimulator

__all__ = [
    "LLMConfig",
    "DEFAULT_LLM_CONFIG",
    "PlanningCondition",
    "LLMSwarmSimulator",
]
