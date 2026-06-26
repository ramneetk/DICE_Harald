"""Reward functional, sync probability, and Fréchet gradients (Sections 4 & 9)."""

from __future__ import annotations

import numpy as np

from .actions import action_power
from .calendar import Calendar
from .config import VPPConfig, DEFAULT_CONFIG


def log_odds(p: float, epsilon: float) -> float:
    p = float(np.clip(p, 0.0, 1.0))
    return np.log((p + epsilon) / (1.0 - p + epsilon + 1e-15))


def sync_probability(
    aggregate_kw: float,
    target_kw: float,
    config: VPPConfig = DEFAULT_CONFIG,
) -> float:
    """
    Soft match score in (0, 1]: 1 when aggregate equals target, decays with |error|.
    Unlike a hard logistic indicator, this retains gradient magnitude when far from target.
    """
    error = abs(aggregate_kw - target_kw)
    tau = max(config.tolerance_tau_kw, 1e-6)
    return float(1.0 / (1.0 + error / tau))


def sync_probability_gradient_wrt_agent(
    agent_probs: np.ndarray,
    neighbor_aggregate_kw: np.ndarray | float,
    target_kw: float,
    config: VPPConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """
    ∇_{π_i} p_sync when total = neighbor_aggregate + agent_power(π_i).
    """
    gamma = np.asarray(config.action_power_kw, dtype=float)
    if isinstance(neighbor_aggregate_kw, np.ndarray):
        # Interval-wise call uses scalar neighbor power per interval
        raise TypeError("neighbor_aggregate_kw must be scalar for gradient")
    agent_kw = float(agent_probs @ gamma)
    total = float(neighbor_aggregate_kw) + agent_kw
    error = abs(total - target_kw)
    tau = max(config.tolerance_tau_kw, 1e-6)
    denom = 1.0 + error / tau
    p = 1.0 / denom
    dp_dtotal = -np.sign(total - target_kw) / (tau * denom ** 2)
    return dp_dtotal * gamma


def log_odds_gradient(
    p: float,
    grad_p: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Eq. (14): G = (1+2ε)/((p+ε)(1-p+ε)) ∇p."""
    denom = (p + epsilon) * (1.0 - p + epsilon)
    scale = (1.0 + 2.0 * epsilon) / max(denom, 1e-12)
    return scale * grad_p


def attractor_gradient(
    total_kw: float,
    target_kw: float,
    config: VPPConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """
    Direct Fréchet gradient of -0.5*(total - target)^2 w.r.t. agent action simplex.
    Used for decentralized PGA coordination (Section 6).
    """
    gamma = np.asarray(config.action_power_kw, dtype=float)
    gap = target_kw - total_kw
    return gap * gamma


def interval_gradients(
    agent_cal: Calendar,
    neighbor_power: np.ndarray,
    target_kw: np.ndarray,
    config: VPPConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """Fréchet gradient G_k for each interval k, shape (E, |Ai|)."""
    E = config.E
    G = np.zeros((E, config.num_actions))
    for k in range(E):
        p = sync_probability(
            neighbor_power[k] + action_power(agent_cal.probs[k], config),
            target_kw[k],
            config,
        )
        grad_p = sync_probability_gradient_wrt_agent(
            agent_cal.probs[k],
            float(neighbor_power[k]),
            float(target_kw[k]),
            config,
        )
        G[k] = log_odds_gradient(p, grad_p, config.epsilon)
    return G


def total_reward(
    agent_cal: Calendar,
    neighbor_power: np.ndarray,
    target_kw: np.ndarray,
    config: VPPConfig = DEFAULT_CONFIG,
) -> float:
    """Discounted integral of log-odds sync reward (Eq. 15–16)."""
    dt = config.dwell_seconds
    gamma_disc = config.discount_gamma
    total = 0.0
    for k in range(E := config.E):
        t_k = k * dt
        t_k1 = (k + 1) * dt
        weight = (np.exp(-gamma_disc * t_k) - np.exp(-gamma_disc * t_k1)) / gamma_disc
        p = sync_probability(
            neighbor_power[k] + action_power(agent_cal.probs[k], config),
            target_kw[k],
            config,
        )
        total += weight * log_odds(p, config.epsilon)
    return float(total)


def lipschitz_bound(config: VPPConfig = DEFAULT_CONFIG, C: float = 1.0) -> float:
    """Proposition 1: Lt <= C * beta^2 / epsilon^2."""
    return C * config.sync_beta ** 2 / config.epsilon ** 2
