"""Tests for steering swarm and experiment matrix."""

from llm_exp.config import FeedbackMode, GuardrailMode, Persona
from llm_exp.steering_swarm import SteeringSwarmSimulator, default_steering_matrix


def test_default_matrix_has_r0_r6():
    matrix = default_steering_matrix()
    ids = [r.run_id for r in matrix]
    assert ids == ["R0", "R1", "R2", "R3", "R4", "R5", "R6"]


def test_steering_swarm_mock_r0_baseline():
    from credit_calculus.config import QUICK_CONFIG
    from dataclasses import replace
    from llm_exp.config import DEFAULT_LLM_CONFIG, SteeringRunConfig

    run = replace(default_steering_matrix()[0], rounds=3)
    sim = SteeringSwarmSimulator(
        run.n,
        run_config=run,
        config=QUICK_CONFIG,
        llm_config=replace(DEFAULT_LLM_CONFIG, use_mock=True),
    )
    metrics = sim.run_steering(max_iter=3)
    assert metrics.run_id == "R0"
    assert metrics.final_tracking_pct < 50.0


def test_steering_swarm_mock_r1_llm():
    from credit_calculus.config import QUICK_CONFIG
    from dataclasses import replace
    from llm_exp.config import DEFAULT_LLM_CONFIG

    run = replace(default_steering_matrix()[1], rounds=2)
    sim = SteeringSwarmSimulator(
        run.n,
        run_config=run,
        config=QUICK_CONFIG,
        llm_config=replace(DEFAULT_LLM_CONFIG, use_mock=True),
    )
    metrics = sim.run_steering(max_iter=2)
    assert metrics.run_id == "R1"
    assert metrics.parse_success_rate == 1.0
    assert metrics.persona == Persona.COOPERATIVE.value
    assert metrics.guardrail_mode == GuardrailMode.FLOOR_ONLY.value
