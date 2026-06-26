"""Swarm coordination via functional projected gradient ascent (Section 6)."""

from __future__ import annotations

import numpy as np

from .actions import IDLE, CHARGE_STD, CHARGE_FAST, DISCHARGE, random_calendar
from .calendar import Calendar, l2_tracking_error, swarm_aggregate_power
from .config import GraphonTopology, VPPConfig, DEFAULT_CONFIG
from .graphon import (
    build_watts_strogatz,
    latency_rbf_weights,
    neighborhood_indices,
    weight_matrix,
    latent_positions,
    delay_matrix,
)
from .guardrail import ascent_projection_guardrail
from .reward import attractor_gradient


def target_attractor(
    E: int,
    n_agents: int,
    config: VPPConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """
    Non-smooth double-step attractor scaled to swarm capacity.
    Paper uses 50 MW with ~10^4 agents; we scale linearly with n.
    """
    max_injection_kw = n_agents * max(config.action_power_kw)  # all DischargeV2G
    max_absorption_kw = n_agents * abs(min(config.action_power_kw))  # all ChargeFast
    scale = n_agents / 10_000.0
    evening_mw = 50_000.0 * scale  # kW
    morning_mw = 50_000.0 * scale

    target = np.zeros(E)
    dt_hours = config.dwell_minutes / 60.0
    for k in range(E):
        hour = (k + 0.5) * dt_hours
        if 17.0 <= hour < 19.0:
            target[k] = min(evening_mw, max_injection_kw * 0.85)
        elif 2.5 <= hour < 4.5:
            target[k] = -min(morning_mw, max_absorption_kw * 0.85)
        else:
            target[k] = 0.0
    return target


def ideal_probs_for_share(share_kw: float, config: VPPConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Deterministic simplex point achieving expected power ≈ share_kw."""
    g = config.action_power_kw
    probs = np.zeros(config.num_actions)
    if abs(share_kw) < 0.05:
        probs[IDLE] = 1.0
    elif share_kw > 0:
        p = min(1.0, share_kw / g[DISCHARGE])
        probs[DISCHARGE] = p
        probs[IDLE] = 1.0 - p
    elif share_kw >= g[CHARGE_STD]:
        p = min(1.0, abs(share_kw) / abs(g[CHARGE_STD]))
        probs[CHARGE_STD] = p
        probs[IDLE] = 1.0 - p
    else:
        p = min(1.0, abs(share_kw) / abs(g[CHARGE_FAST]))
        probs[CHARGE_FAST] = p
        probs[IDLE] = 1.0 - p
    return probs


def per_agent_target(total_target: np.ndarray, n: int) -> np.ndarray:
    return total_target / max(n, 1)


class SwarmSimulator:
    def __init__(
        self,
        n: int,
        config: VPPConfig = DEFAULT_CONFIG,
        topology: GraphonTopology = GraphonTopology.FLAT,
        rng: np.random.Generator | None = None,
    ):
        self.n = n
        self.config = config
        self.topology = topology
        self.rng = rng or np.random.default_rng(42)
        self.graph = build_watts_strogatz(n, config)
        self.alphas = latent_positions(n, self.rng)
        self.W = weight_matrix(self.alphas, topology)
        self.delays = delay_matrix(n, self.rng)
        self.W_rbf = latency_rbf_weights(self.W, self.delays, config.graphon_decay)
        self.calendars = [Calendar.idle(config) for _ in range(n)]
        self.global_target = target_attractor(config.E, n, config)
        self.agent_types = np.zeros(n, dtype=int)

    def neighbor_power(self, i: int) -> np.ndarray:
        Ki = neighborhood_indices(self.graph, i, self.config.neighborhood_size)
        power = np.zeros(self.config.E)
        for j in Ki:
            if j != i:
                power += self.W_rbf[i, j] * self.calendars[j].power_profile_kw()
        return power

    def aggregate(self) -> np.ndarray:
        return swarm_aggregate_power(self.calendars)

    def all_powers_matrix(self) -> np.ndarray:
        """Shape (n, E) expected kW per agent per interval."""
        return np.stack([c.power_profile_kw() for c in self.calendars], axis=0)

    def neighbor_power_batch(self) -> np.ndarray:
        """Vectorized neighbor power for all agents, shape (n, E)."""
        P = self.all_powers_matrix()
        n, E = P.shape
        out = np.zeros((n, E))
        for i in range(n):
            Ki = neighborhood_indices(self.graph, i, self.config.neighborhood_size)
            for j in Ki:
                if j != i:
                    out[i] += self.W_rbf[i, j] * P[j]
        return out

    def tracking_error_pct(self) -> float:
        agg = self.aggregate()
        err = l2_tracking_error(agg, self.global_target, self.config)
        scale = max(
            float(np.linalg.norm(self.global_target) * np.sqrt(self.config.dwell_seconds)),
            1.0,
        )
        return 100.0 * err / scale

    def step_pga(self, eta: float | None = None) -> None:
        """Functional PGA: blend calendars toward fair-share attractor."""
        agent_share = self.global_target / max(self.n, 1)
        blend = 0.35 if eta is None else float(np.clip(eta, 0.01, 1.0))

        new_calendars = []
        for i in range(self.n):
            if self.agent_types[i] == 1:
                new_calendars.append(self.calendars[i].copy())
                continue

            updated = self.calendars[i].probs.copy()
            for k in range(self.config.E):
                ideal = ideal_probs_for_share(float(agent_share[k]), self.config)
                updated[k] = (1.0 - blend) * updated[k] + blend * ideal
                updated[k] /= max(updated[k].sum(), 1e-12)

            new_calendars.append(Calendar(updated, self.config))

        self.calendars = new_calendars

    def run_until_converged(self, max_iter: int | None = None) -> dict:
        max_iter = max_iter or self.config.max_iterations
        errors = []
        for it in range(max_iter):
            prev = self.aggregate().copy()
            self.step_pga()
            err = self.tracking_error_pct()
            errors.append(err)
            delta = float(np.max(np.abs(self.aggregate() - prev)))
            if delta < self.config.convergence_tol and err < 5.0:
                return {"iterations": it + 1, "errors": errors, "converged": True}
        return {"iterations": max_iter, "errors": errors, "converged": False}

    def apply_churn(self, fraction: float) -> None:
        n_churn = int(fraction * self.n)
        if n_churn == 0:
            return
        nodes = self.rng.choice(self.n, size=n_churn, replace=False)
        for node in nodes:
            others = [j for j in range(self.n) if j != node]
            new_nbrs = self.rng.choice(
                others, size=min(self.config.ws_k, len(others)), replace=False
            )
            self.graph.remove_edges_from(list(self.graph.edges(node)))
            for nb in new_nbrs:
                self.graph.add_edge(node, int(nb))

    def mark_byzantine(self, fraction: float) -> None:
        n_bad = int(fraction * self.n)
        idx = self.rng.choice(self.n, size=n_bad, replace=False)
        self.agent_types[:] = 0
        self.agent_types[idx] = 1
        for i in idx:
            probs = np.zeros((self.config.E, self.config.num_actions))
            probs[:, 0] = 1.0
            for k in range(self.config.E):
                hour = (k + 0.5) * self.config.dwell_minutes / 60.0
                if 17.0 <= hour < 19.0:
                    probs[k, 3] = 1.0
            self.calendars[i] = Calendar(probs, self.config)

    def mark_schemers(self, fraction: float) -> None:
        n_schem = int(fraction * self.n)
        available = np.where(self.agent_types == 0)[0]
        idx = self.rng.choice(available, size=min(n_schem, len(available)), replace=False)
        self.agent_types[idx] = 2
