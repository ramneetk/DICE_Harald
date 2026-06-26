"""Tests for LLM calendar parser (no GPU)."""

import json

import numpy as np

from credit_calculus.config import QUICK_CONFIG
from llm_exp.parser import events_to_calendar, parse_llm_calendar
from llm_exp.prompts import build_user_prompt, calendar_to_events, FEW_SHOT_EXAMPLE


def test_parse_valid_json():
    text = json.dumps(FEW_SHOT_EXAMPLE)
    result = parse_llm_calendar(text, QUICK_CONFIG)
    assert result.success
    assert result.calendar.probs.shape == (QUICK_CONFIG.E, QUICK_CONFIG.num_actions)


def test_parse_fenced_json():
    text = "```json\n" + json.dumps(FEW_SHOT_EXAMPLE) + "\n```"
    result = parse_llm_calendar(text, QUICK_CONFIG)
    assert result.success


def test_parse_invalid_fallback():
    result = parse_llm_calendar("not json at all", QUICK_CONFIG)
    assert not result.success
    assert np.allclose(result.calendar.probs[:, 0], 1.0)


def test_events_to_calendar_evening_discharge():
    events = [
        {"start_hour": 0.0, "action": "Idle"},
        {"start_hour": 17.0, "action": "DischargeV2G"},
    ]
    cal = events_to_calendar(events, QUICK_CONFIG)
    power = cal.power_profile_kw()
    assert max(power) > 0


def test_build_user_prompt():
    E = QUICK_CONFIG.E
    target = np.zeros(E)
    agg = np.zeros(E)
    share = np.zeros(E)
    prompt = build_user_prompt(0, 0, target, agg, share, QUICK_CONFIG)
    assert "agent_id" in prompt
    assert "example_output" in prompt


def test_calendar_roundtrip():
    events = FEW_SHOT_EXAMPLE["events"]
    cal = events_to_calendar(events, QUICK_CONFIG)
    back = calendar_to_events(cal.probs, QUICK_CONFIG)
    assert len(back) >= 1
    assert back[0]["action"] == "Idle"
