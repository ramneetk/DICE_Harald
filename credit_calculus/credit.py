"""Epistemic planning credit and Shapley-style allocation (Sections 4–5)."""

from __future__ import annotations

import numpy as np

from .calendar import Calendar
from .config import VPPConfig, DEFAULT_CONFIG
from .reward import total_reward


def counterfactual_credit(
    agent_cal: Calendar,
    neighbor_power: np.ndarray,
    target_kw: np.ndarray,
    config: VPPConfig = DEFAULT_CONFIG,
) -> float:
    """Eq. (19): credit vs idle baseline."""
    cooperative = total_reward(agent_cal, neighbor_power, target_kw, config)
    idle = total_reward(Calendar.idle(config), neighbor_power, target_kw, config)
    return cooperative - idle


def approximate_shapley_credit(
    agent_idx: int,
    calendars: list[Calendar],
    neighbor_indices: list[int],
    target_kw: np.ndarray,
    config: VPPConfig = DEFAULT_CONFIG,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Monte Carlo Shapley estimate over sub-coalitions in Ki.
    For large |Ki| we sample permutations (O(m) samples).
    """
    if rng is None:
        rng = np.random.default_rng(agent_idx)

    Ki = [j for j in neighbor_indices if j != agent_idx]
    m = len(Ki)
    if m == 0:
        return counterfactual_credit(
            calendars[agent_idx],
            np.zeros(config.E),
            target_kw,
            config,
        )

    n_samples = min(64, max(8, 2 * m))
    marginal_sum = 0.0

    for _ in range(n_samples):
        perm = rng.permutation(Ki)
        # Build neighbor power without agent i
        base_power = np.zeros(config.E)
        for j in perm:
            base_power += calendars[j].power_profile_kw()

        v_empty = total_reward(Calendar.idle(config), base_power, target_kw, config)
        v_with = total_reward(calendars[agent_idx], base_power, target_kw, config)
        marginal_sum += v_with - v_empty

        # Incremental coalitions along permutation
        coalition_power = base_power.copy()
        for j in perm:
            v_before = total_reward(Calendar.idle(config), coalition_power, target_kw, config)
            coalition_power += calendars[j].power_profile_kw()
            v_after = total_reward(Calendar.idle(config), coalition_power, target_kw, config)
            if j == agent_idx:
                marginal_sum += v_after - v_before

    # Simpler: use counterfactual as dominant term for scalability
    cf = counterfactual_credit(
        calendars[agent_idx],
        _neighbor_power_without(agent_idx, calendars, neighbor_indices),
        target_kw,
        config,
    )
    shapley_est = cf + marginal_sum / max(n_samples * (m + 1), 1)
    return float(shapley_est)


def _neighbor_power_without(
    agent_idx: int,
    calendars: list[Calendar],
    neighbor_indices: list[int],
) -> np.ndarray:
    E = calendars[0].config.E
    power = np.zeros(E)
    for j in neighbor_indices:
        if j != agent_idx:
            power += calendars[j].power_profile_kw()
    return power
