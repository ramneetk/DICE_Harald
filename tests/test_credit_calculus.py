"""Basic tests for Credit Calculus simulation."""

import numpy as np

from credit_calculus.config import DEFAULT_CONFIG, QUICK_CONFIG
from credit_calculus.guardrail import project_simplex, benchmark_guardrail_ms
from credit_calculus.adversarial import (
    deviation_penalty,
    realized_payout,
    schemer_published_calendar,
    schemer_executed_calendar,
)
from credit_calculus.swarm import SwarmSimulator, ideal_probs_for_share


def test_simplex_projection():
    v = np.array([0.3, 0.9, -0.2])
    p = project_simplex(v)
    assert abs(p.sum() - 1.0) < 1e-9
    assert np.all(p >= -1e-12)


def test_schemer_slashing():
    pub = schemer_published_calendar(QUICK_CONFIG)
    exe = schemer_executed_calendar(QUICK_CONFIG)
    pen = deviation_penalty(pub, exe, QUICK_CONFIG)
    assert pen > 0
    assert realized_payout(5.0, pen, QUICK_CONFIG) == 0.0


def test_swarm_convergence():
    sim = SwarmSimulator(200, QUICK_CONFIG)
    result = sim.run_until_converged(max_iter=20)
    assert sim.tracking_error_pct() < 5.0
    assert result["iterations"] <= 20


def test_ideal_share_power():
    cfg = QUICK_CONFIG
    p = ideal_probs_for_share(5.0, cfg)
    assert abs(p @ np.array(cfg.action_power_kw) - 5.0) < 0.01


def test_guardrail_latency_order():
    ms = benchmark_guardrail_ms(DEFAULT_CONFIG.E, DEFAULT_CONFIG.num_actions, DEFAULT_CONFIG)
    assert ms < 100.0  # sub-100ms per agent on typical hardware
