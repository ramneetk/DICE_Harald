"""Action space and power mapping."""

from __future__ import annotations

import numpy as np

from .config import VPPConfig, DEFAULT_CONFIG

IDLE, CHARGE_STD, CHARGE_FAST, DISCHARGE = 0, 1, 2, 3


def action_power(probs: np.ndarray, config: VPPConfig = DEFAULT_CONFIG) -> float:
    """Expected kW injection from a probability vector over actions."""
    gamma = np.asarray(config.action_power_kw, dtype=float)
    return float(probs @ gamma)


def hard_action(probs: np.ndarray) -> int:
    return int(np.argmax(probs))


def one_hot_action(action_idx: int, num_actions: int) -> np.ndarray:
    v = np.zeros(num_actions, dtype=float)
    v[action_idx] = 1.0
    return v


def random_calendar(
    E: int,
    num_actions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return (E, num_actions) simplex rows."""
    raw = rng.dirichlet(np.ones(num_actions), size=E)
    return raw.astype(float)
