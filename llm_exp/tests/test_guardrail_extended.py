"""Tests for extended guardrail modes."""

import numpy as np

from credit_calculus.calendar import Calendar
from credit_calculus.config import QUICK_CONFIG
from credit_calculus.guardrail_extended import (
    apply_extended_guardrail,
    extended_guardrail,
    schedule_utility,
)


def _random_calendar(rng: np.random.Generator) -> Calendar:
    cfg = QUICK_CONFIG
    probs = rng.dirichlet(np.ones(cfg.num_actions), size=cfg.E)
    return Calendar(probs, cfg)


def test_floor_only_passes_high_utility():
    cfg = QUICK_CONFIG
    rng = np.random.default_rng(0)
    cal = _random_calendar(rng)
    nbr = np.zeros(cfg.E)
    target = np.zeros(cfg.E)
    G = np.ones((cfg.E, cfg.num_actions)) * 0.1
    u = schedule_utility(cal.probs, G, cfg)
    projected, info = extended_guardrail(
        cal.probs, G, floor_b=u - 1.0, mode="floor_only", config=cfg
    )
    assert not info.get("modified", True)


def test_attractor_seek_modifies_low_utility():
    cfg = QUICK_CONFIG
    idle = Calendar.idle(cfg)
    nbr = np.zeros(cfg.E)
    target = np.ones(cfg.E) * 10.0
    share = target / 5
    projected, info = apply_extended_guardrail(
        idle,
        nbr,
        target,
        mode="attractor_seek",
        config=cfg,
        agent_share=share,
    )
    assert projected.probs.shape == idle.probs.shape
    assert info["mode"] == "attractor_seek"


def test_hybrid_band_runs():
    cfg = QUICK_CONFIG
    rng = np.random.default_rng(42)
    cal = _random_calendar(rng)
    nbr = np.zeros(cfg.E)
    target = np.zeros(cfg.E)
    projected, info = apply_extended_guardrail(
        cal,
        nbr,
        target,
        mode="hybrid_band",
        config=cfg,
        agent_share=target,
    )
    assert info["mode"] == "hybrid_band"
    assert projected.probs.shape == cal.probs.shape


def test_none_mode_passthrough():
    cfg = QUICK_CONFIG
    cal = Calendar.idle(cfg)
    nbr = np.zeros(cfg.E)
    target = np.zeros(cfg.E)
    projected, info = apply_extended_guardrail(
        cal, nbr, target, mode="none", config=cfg
    )
    assert not info["modified"]
    np.testing.assert_allclose(projected.probs, cal.probs)
