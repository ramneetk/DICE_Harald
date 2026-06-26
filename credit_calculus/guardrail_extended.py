"""Extended guardrail modes: attractor-seeking and hybrid floor/ceiling band."""

from __future__ import annotations

import numpy as np

from .calendar import Calendar
from .config import VPPConfig, DEFAULT_CONFIG
from .guardrail import ascent_projection_guardrail, project_simplex
from .reward import interval_gradients
from .swarm import ideal_probs_for_share, per_agent_target


def schedule_utility(
    probs: np.ndarray,
    gradients: np.ndarray,
    config: VPPConfig = DEFAULT_CONFIG,
) -> float:
    dt = config.dwell_seconds
    return float(sum(dt * float(gradients[k] @ probs[k]) for k in range(config.E)))


def ideal_utility(
    agent_share: np.ndarray,
    gradients: np.ndarray,
    config: VPPConfig = DEFAULT_CONFIG,
) -> float:
    """Utility of fair-share ideal calendar at each interval."""
    dt = config.dwell_seconds
    total = 0.0
    for k in range(config.E):
        ideal = ideal_probs_for_share(float(agent_share[k]), config)
        total += dt * float(gradients[k] @ ideal)
    return float(total)


def descent_projection_guardrail(
    pi_llm: np.ndarray,
    gradients: np.ndarray,
    ceiling_c: float,
    config: VPPConfig = DEFAULT_CONFIG,
) -> tuple[np.ndarray, dict]:
    """Project schedule downward when utility exceeds ceiling (descent guardrail)."""
    E, _ = pi_llm.shape
    dt = config.dwell_seconds
    tol = config.bisection_tol

    utility_prior = schedule_utility(pi_llm, gradients, config)
    if utility_prior <= ceiling_c:
        return pi_llm.copy(), {
            "modified": False,
            "mu": 0.0,
            "bisection_iters": 0,
            "utility": utility_prior,
            "hit_ceiling": False,
        }

    g_norm_sq = sum(dt * float(np.linalg.norm(gradients[k]) ** 2) for k in range(E))
    if g_norm_sq < 1e-15:
        return pi_llm.copy(), {
            "modified": False,
            "mu": 0.0,
            "bisection_iters": 0,
            "utility": utility_prior,
            "hit_ceiling": False,
        }

    mu_low = 0.0
    mu_high = max(0.0, (utility_prior - ceiling_c) / g_norm_sq)
    bisection_iters = 0

    def constraint_value(mu: float) -> float:
        total = 0.0
        for k in range(E):
            shifted = project_simplex(pi_llm[k] - mu * gradients[k])
            total += dt * float(gradients[k] @ shifted)
        return total

    while mu_high - mu_low > tol:
        mu_mid = 0.5 * (mu_low + mu_high)
        if constraint_value(mu_mid) <= ceiling_c:
            mu_high = mu_mid
        else:
            mu_low = mu_mid
        bisection_iters += 1

    pi_star = np.zeros_like(pi_llm)
    for k in range(E):
        pi_star[k] = project_simplex(pi_llm[k] - mu_high * gradients[k])

    final_utility = constraint_value(mu_high)
    return pi_star, {
        "modified": True,
        "mu": mu_high,
        "bisection_iters": bisection_iters,
        "utility": final_utility,
        "hit_ceiling": True,
    }


def extended_guardrail(
    pi_llm: np.ndarray,
    gradients: np.ndarray,
    floor_b: float,
    mode: str,
    config: VPPConfig = DEFAULT_CONFIG,
    agent_share: np.ndarray | None = None,
    attractor_target_fraction: float = 0.9,
    ceiling_fraction: float = 1.2,
) -> tuple[np.ndarray, dict]:
    """
    Apply guardrail mode: floor_only | attractor_seek | hybrid_band | none.

    Returns projected schedule and diagnostics.
    """
    if mode == "none":
        utility = schedule_utility(pi_llm, gradients, config)
        return pi_llm.copy(), {
            "modified": False,
            "mode": mode,
            "utility": utility,
            "floor_b": floor_b,
        }

    utility_prior = schedule_utility(pi_llm, gradients, config)
    info_base: dict = {
        "mode": mode,
        "utility_prior": utility_prior,
        "floor_b": floor_b,
        "hit_floor": False,
        "hit_ceiling": False,
    }

    if mode == "floor_only":
        projected, info = ascent_projection_guardrail(pi_llm, gradients, floor_b, config)
        info.update(info_base)
        info["hit_floor"] = info.get("modified", False)
        return projected, info

    if mode == "attractor_seek":
        target = floor_b
        if agent_share is not None:
            ideal_u = ideal_utility(agent_share, gradients, config)
            target = max(floor_b, attractor_target_fraction * ideal_u)
        projected, info = ascent_projection_guardrail(
            pi_llm, gradients, target, config
        )
        info.update(info_base)
        info["target_utility"] = target
        info["hit_floor"] = info.get("modified", False)
        return projected, info

    if mode == "hybrid_band":
        ceiling_c = ceiling_fraction * max(floor_b, 1e-12)
        if agent_share is not None:
            ideal_u = ideal_utility(agent_share, gradients, config)
            ceiling_c = max(ceiling_c, ceiling_fraction * ideal_u)

        pi_work = pi_llm.copy()
        hit_ceiling = False
        hit_floor = False

        u = schedule_utility(pi_work, gradients, config)
        if u > ceiling_c:
            pi_work, info_down = descent_projection_guardrail(
                pi_work, gradients, ceiling_c, config
            )
            hit_ceiling = info_down.get("modified", False)
            info_base.update(info_down)

        u = schedule_utility(pi_work, gradients, config)
        if u < floor_b:
            pi_work, info_up = ascent_projection_guardrail(
                pi_work, gradients, floor_b, config
            )
            hit_floor = info_up.get("modified", False)
            info_base.update(info_up)

        final_u = schedule_utility(pi_work, gradients, config)
        info_base.update({
            "modified": hit_ceiling or hit_floor,
            "utility": final_u,
            "ceiling_c": ceiling_c,
            "hit_ceiling": hit_ceiling,
            "hit_floor": hit_floor,
        })
        return pi_work, info_base

    raise ValueError(f"Unknown guardrail mode: {mode}")


def apply_extended_guardrail(
    calendar: Calendar,
    neighbor_power: np.ndarray,
    target_kw: np.ndarray,
    mode: str,
    config: VPPConfig = DEFAULT_CONFIG,
    agent_share: np.ndarray | None = None,
) -> tuple[Calendar, dict]:
    """Project calendar using extended guardrail mode."""
    G = interval_gradients(calendar, neighbor_power, target_kw, config)
    if agent_share is None:
        agent_share = per_agent_target(target_kw, 1)
    floor_b = config.cooperative_floor_fraction * float(
        np.mean([G[k] @ calendar.probs[k] for k in range(config.E)])
    )
    projected, info = extended_guardrail(
        calendar.probs,
        G,
        floor_b,
        mode,
        config,
        agent_share=agent_share,
    )
    return Calendar(projected, config), info
