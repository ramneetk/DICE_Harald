"""Mock LLM client and swarm smoke tests."""

from dataclasses import replace

from credit_calculus.config import QUICK_CONFIG
from llm_exp.client import MockLLMClient
from llm_exp.config import DEFAULT_LLM_CONFIG, PlanningCondition
from llm_exp.swarm import LLMSwarmSimulator


def test_mock_client_returns_json():
    client = MockLLMClient()
    resp = client.complete("sys", "user")
    assert "events" in resp.text


def test_llm_swarm_mock_guarded(tmp_path):
    llm_cfg = replace(DEFAULT_LLM_CONFIG, use_mock=True)
    log = tmp_path / "test.jsonl"
    sim = LLMSwarmSimulator(
        5,
        condition=PlanningCondition.LLM_GUARDED,
        config=QUICK_CONFIG,
        llm_config=llm_cfg,
        log_path=log,
    )
    metrics = sim.run_until_converged(max_iter=2)
    assert metrics.parse_success_rate == 1.0
    assert len(metrics.errors) == 2
    assert log.exists()


def test_llm_swarm_baseline():
    llm_cfg = replace(DEFAULT_LLM_CONFIG, use_mock=True)
    sim = LLMSwarmSimulator(
        10,
        condition=PlanningCondition.BASELINE,
        config=QUICK_CONFIG,
        llm_config=llm_cfg,
    )
    metrics = sim.run_until_converged(max_iter=3)
    assert metrics.condition == PlanningCondition.BASELINE
