"""LLM experiment configuration."""

from dataclasses import dataclass
from enum import Enum


class PlanningCondition(str, Enum):
    BASELINE = "baseline"
    LLM_RAW = "llm_raw"
    LLM_GUARDED = "llm_guarded"


@dataclass(frozen=True)
class LLMConfig:
    model: str = "Qwen/Qwen3.5-0.8B"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    temperature: float = 0.2
    max_tokens: int = 512
    timeout_s: float = 120.0
    max_retries: int = 1
    seed: int = 42
    use_mock: bool = False


DEFAULT_LLM_CONFIG = LLMConfig()
