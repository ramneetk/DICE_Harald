"""LLM experiment configuration."""

from dataclasses import dataclass
from enum import Enum


class PlanningCondition(str, Enum):
    BASELINE = "baseline"
    LLM_RAW = "llm_raw"
    LLM_GUARDED = "llm_guarded"


class Persona(str, Enum):
    CONSERVATIVE = "conservative"
    COOPERATIVE = "cooperative"
    AGGRESSIVE = "aggressive"
    MISSION_CRITICAL = "mission_critical"
    BYZANTINE = "byzantine"


class GuardrailMode(str, Enum):
    NONE = "none"
    FLOOR_ONLY = "floor_only"
    ATTRACTOR_SEEK = "attractor_seek"
    HYBRID_BAND = "hybrid_band"


class FeedbackMode(str, Enum):
    NONE = "none"
    CREDIT_ONLY = "credit_only"
    GRADIENT_ONLY = "gradient_only"
    CREDIT_GRADIENT = "credit_gradient"
    FULL = "full"


@dataclass(frozen=True)
class SteeringRunConfig:
    """Single run in the steering experiment matrix."""

    run_id: str
    persona: Persona
    guardrail_mode: GuardrailMode
    feedback_mode: FeedbackMode
    n: int
    rounds: int
    use_baseline: bool = False


@dataclass(frozen=True)
class LLMConfig:
    model: str = "Qwen/Qwen3.5-0.8B"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    temperature: float = 0.0
    max_tokens: int = 384
    timeout_s: float = 120.0
    max_retries: int = 1
    seed: int = 42
    use_mock: bool = False
    use_json_mode: bool = True


DEFAULT_LLM_CONFIG = LLMConfig()
