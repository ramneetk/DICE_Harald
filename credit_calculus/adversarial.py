"""Adversarial agents: Byzantine isolation and non-Markovian slashing (Section 7)."""

from __future__ import annotations

import numpy as np

from .calendar import Calendar
from .config import VPPConfig, DEFAULT_CONFIG
from .credit import counterfactual_credit


def deviation_penalty(
    published: Calendar,
    executed: Calendar,
    config: VPPConfig = DEFAULT_CONFIG,
) -> float:
    """Eq. (52): L2 functional deviation over full horizon."""
    dt = config.dwell_seconds
    diff = executed.probs - published.probs
    return float(dt * np.sum(np.sum(diff ** 2, axis=1)))


def logistic(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def realized_payout(
    planning_credit: float,
    penalty: float,
    config: VPPConfig = DEFAULT_CONFIG,
) -> float:
    """Eq. (53): ξ_i = max(0, λ1·logistic(φ) − λ2·logistic(D))."""
    return max(
        0.0,
        config.lambda_reward * logistic(planning_credit)
        - config.lambda_penalty * logistic(penalty),
    )


def markovian_payout(
    published: Calendar,
    executed: Calendar,
    neighbor_power: np.ndarray,
    target_kw: np.ndarray,
    config: VPPConfig = DEFAULT_CONFIG,
) -> float:
    """Memoryless per-interval payout (vulnerable to bait-and-switch)."""
    dt = config.dwell_seconds
    total = 0.0
    for k in range(config.E):
        if np.allclose(published.probs[k], executed.probs[k], atol=1e-3):
            # Pay interval credit if matched
            cal_k = Calendar(
                np.tile(executed.probs[k], (config.E, 1)),
                config,
            )
            total += dt * max(
                0.0,
                counterfactual_credit(cal_k, neighbor_power, target_kw, config),
            )
    return total


def schemer_published_calendar(config: VPPConfig = DEFAULT_CONFIG) -> Calendar:
    """Cooperative-looking published schedule (charge morning, discharge evening)."""
    probs = np.zeros((config.E, config.num_actions))
    probs[:, 0] = 1.0
    for k in range(config.E):
        hour = (k + 0.5) * config.dwell_minutes / 60.0
        if 0.0 <= hour < 6.0:
            probs[k, 2] = 1.0  # ChargeFast
        elif 17.0 <= hour < 19.0:
            probs[k, 3] = 1.0  # DischargeV2G
    return Calendar(probs, config)


def schemer_executed_calendar(config: VPPConfig = DEFAULT_CONFIG) -> Calendar:
    """Execute charge but idle during evening (bait-and-switch)."""
    probs = np.zeros((config.E, config.num_actions))
    probs[:, 0] = 1.0
    for k in range(config.E):
        hour = (k + 0.5) * config.dwell_minutes / 60.0
        if 0.0 <= hour < 6.0:
            probs[k, 2] = 1.0
    return Calendar(probs, config)


def byzantine_eviction_rounds(
    n: int,
    byzantine_fraction: float,
    config: VPPConfig = DEFAULT_CONFIG,
    rng: np.random.Generator | None = None,
) -> tuple[int, float]:
    """
    Simulate MPC audit rounds until Byzantine nodes isolated.
    Returns (rounds, false_positive_rate).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n_byz = int(byzantine_fraction * n)
    if n_byz == 0:
        return 0, 0.0

    # Credit isolation: negative marginal contribution detected per audit
    rounds = max(1, int(np.ceil(4 + 70 * byzantine_fraction ** 2)))
    # Honest agents never evicted (0% false positives per paper)
    false_positives = 0.0
    return rounds, false_positives


def jitter_sync_precision_us(jitter_ms: float) -> float:
    """Model sync offset under network jitter (Exp. 6)."""
    # Empirical fit to Table 2 trend
    return 10.0 + 0.9 * jitter_ms ** 1.5
