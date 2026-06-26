"""Algorithm 1: Asynchronous Ascent Projection Guardrail (Section 9)."""

from __future__ import annotations

import time

import numpy as np

from .config import VPPConfig, DEFAULT_CONFIG


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Project vector onto probability simplex (Duchi et al. sorting method)."""
    v = np.asarray(v, dtype=float)
    if v.ndim != 1:
        raise ValueError("project_simplex expects 1D vector")
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
    if rho.size == 0:
        lam = (np.sum(u) - 1.0) / n
    else:
        rho_max = rho[-1]
        lam = (cssv[rho_max] - 1.0) / (rho_max + 1)
    w = np.maximum(v - lam, 0.0)
    s = w.sum()
    if s <= 0:
        w = np.ones(n) / n
    else:
        w /= s
    return w


def ascent_projection_guardrail(
    pi_llm: np.ndarray,
    gradients: np.ndarray,
    floor_b: float,
    config: VPPConfig = DEFAULT_CONFIG,
) -> tuple[np.ndarray, dict]:
    """
    Project piecewise schedule onto cooperative manifold (Algorithm 1).

    pi_llm, gradients: shape (E, |Ai|)
    Returns projected schedule and diagnostics.
    """
    E, num_actions = pi_llm.shape
    dt = config.dwell_seconds
    tol = config.bisection_tol

    utility_prior = sum(
        dt * float(gradients[k] @ pi_llm[k]) for k in range(E)
    )
    if utility_prior >= floor_b:
        return pi_llm.copy(), {
            "modified": False,
            "mu": 0.0,
            "bisection_iters": 0,
            "utility": utility_prior,
        }

    g_norm_sq = sum(dt * float(np.linalg.norm(gradients[k]) ** 2) for k in range(E))
    if g_norm_sq < 1e-15:
        return pi_llm.copy(), {
            "modified": False,
            "mu": 0.0,
            "bisection_iters": 0,
            "utility": utility_prior,
        }

    mu_low = 0.0
    mu_high = max(0.0, (floor_b - utility_prior) / g_norm_sq)
    bisection_iters = 0

    def constraint_value(mu: float) -> float:
        total = 0.0
        for k in range(E):
            shifted = project_simplex(pi_llm[k] + mu * gradients[k])
            total += dt * float(gradients[k] @ shifted)
        return total

    while mu_high - mu_low > tol:
        mu_mid = 0.5 * (mu_low + mu_high)
        if constraint_value(mu_mid) >= floor_b:
            mu_high = mu_mid
        else:
            mu_low = mu_mid
        bisection_iters += 1

    pi_star = np.zeros_like(pi_llm)
    for k in range(E):
        pi_star[k] = project_simplex(pi_llm[k] + mu_high * gradients[k])

    final_utility = constraint_value(mu_high)
    return pi_star, {
        "modified": True,
        "mu": mu_high,
        "bisection_iters": bisection_iters,
        "utility": final_utility,
    }


def benchmark_guardrail_ms(
    E: int,
    num_actions: int,
    config: VPPConfig = DEFAULT_CONFIG,
    rng: np.random.Generator | None = None,
) -> float:
    """Time one guardrail invocation (per-agent latency, ms)."""
    if rng is None:
        rng = np.random.default_rng(0)
    pi = rng.dirichlet(np.ones(num_actions), size=E)
    G = rng.normal(size=(E, num_actions))
    floor_b = config.cooperative_floor_fraction * float(
        np.mean([G[k] @ pi[k] for k in range(E)])
    )

    # Warm-up
    ascent_projection_guardrail(pi, G, floor_b, config)

    n_runs = 50
    t0 = time.perf_counter()
    for _ in range(n_runs):
        ascent_projection_guardrail(pi, G, floor_b, config)
    elapsed_ms = (time.perf_counter() - t0) / n_runs * 1000.0
    return elapsed_ms
