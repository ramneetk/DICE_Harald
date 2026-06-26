"""Tests for feedback prompt blocks."""

import numpy as np

from credit_calculus.config import QUICK_CONFIG
from llm_exp.config import FeedbackMode
from llm_exp.prompts import FeedbackContext, build_user_prompt, gradient_hints_from_gradients


def test_build_user_prompt_no_feedback():
    cfg = QUICK_CONFIG
    E = cfg.E
    target = np.zeros(E)
    target[10] = 50.0
    aggregate = np.zeros(E)
    share = target / 10
    prompt = build_user_prompt(0, 1, target, aggregate, share, cfg)
    assert "Agent 0" in prompt
    assert "Shapley" not in prompt
    assert "gradient hints" not in prompt


def test_build_user_prompt_credit_feedback():
    cfg = QUICK_CONFIG
    E = cfg.E
    target = np.zeros(E)
    aggregate = np.zeros(E)
    share = target / 10
    feedback = FeedbackContext(
        mode=FeedbackMode.CREDIT_ONLY,
        credit=3.14,
        rank=2,
        swarm_size=10,
    )
    prompt = build_user_prompt(1, 2, target, aggregate, share, cfg, feedback=feedback)
    assert "Shapley credit: 3.14" in prompt
    assert "Rank: 2/10" in prompt


def test_build_user_prompt_full_feedback():
    cfg = QUICK_CONFIG
    E = cfg.E
    target = np.zeros(E)
    aggregate = np.zeros(E)
    share = target / 10
    feedback = FeedbackContext(
        mode=FeedbackMode.FULL,
        credit=1.0,
        rank=1,
        swarm_size=5,
        penalty=0.5,
        payout=0.8,
        gradient_hints=[{"hour": 17.0, "action": "DischargeV2G"}],
    )
    prompt = build_user_prompt(0, 0, target, aggregate, share, cfg, feedback=feedback)
    assert "Shapley credit" in prompt
    assert "gradient hints" in prompt
    assert "Deviation penalty" in prompt
    assert "Non-Markovian payout" in prompt
    assert "17.0h→DischargeV2G" in prompt


def test_gradient_hints_from_gradients():
    cfg = QUICK_CONFIG
    G = np.zeros((cfg.E, cfg.num_actions))
    G[5, 3] = 2.0
    hints = gradient_hints_from_gradients(G, cfg, max_hints=2)
    assert len(hints) >= 1
    assert hints[0]["action"] == "DischargeV2G"
