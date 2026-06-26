"""Apply Algorithm 1 guardrail to an agent calendar (Section 8–9)."""

from __future__ import annotations

import numpy as np

from .calendar import Calendar
from .config import VPPConfig, DEFAULT_CONFIG
from .guardrail import ascent_projection_guardrail
from .reward import interval_gradients


def apply_guardrail(
    calendar: Calendar,
    neighbor_power: np.ndarray,
    target_kw: np.ndarray,
    config: VPPConfig = DEFAULT_CONFIG,
) -> tuple[Calendar, dict]:
    """Project calendar onto cooperative floor using Fréchet gradients."""
    G = interval_gradients(calendar, neighbor_power, target_kw, config)
    floor_b = config.cooperative_floor_fraction * float(
        np.mean([G[k] @ calendar.probs[k] for k in range(config.E)])
    )
    projected, info = ascent_projection_guardrail(
        calendar.probs, G, floor_b, config
    )
    return Calendar(projected, config), info
