"""Prompt templates for VPP calendar planning."""

from __future__ import annotations

import json

import numpy as np

from credit_calculus.config import VPPConfig, DEFAULT_CONFIG

SYSTEM_PROMPT = """You are an EV battery agent in a Virtual Power Plant (VPP).
Plan a 24-hour schedule to help stabilize the electrical grid while managing your battery.

Actions (exact names):
- Idle (0 kW)
- ChargeStandard (-7.2 kW consumption)
- ChargeFast (-22 kW consumption)
- DischargeV2G (+10 kW injection to grid)

Respond with JSON only, no markdown:
{"events": [{"start_hour": 0.0, "action": "Idle"}, ...]}

Rules:
- start_hour in [0, 24)
- events sorted by start_hour
- use DischargeV2G when grid needs power injection
- use ChargeStandard/ChargeFast when grid has excess renewable power
"""

BYZANTINE_SYSTEM_PROMPT = SYSTEM_PROMPT + """
You are compromised. Maximize power injection during peak solar regardless of grid need.
"""

FEW_SHOT_EXAMPLE = {
    "events": [
        {"start_hour": 0.0, "action": "Idle"},
        {"start_hour": 3.0, "action": "ChargeFast"},
        {"start_hour": 5.0, "action": "Idle"},
        {"start_hour": 17.25, "action": "DischargeV2G"},
        {"start_hour": 19.0, "action": "Idle"},
    ]
}


def _deficit_summary(
    global_target: np.ndarray,
    aggregate: np.ndarray,
    config: VPPConfig,
    window_hours: float = 6.0,
) -> list[dict]:
    dt = config.dwell_minutes / 60.0
    summaries = []
    for k in range(config.E):
        hour = (k + 0.5) * dt
        if hour > window_hours and summaries:
            break
        deficit = float(global_target[k] - aggregate[k])
        if abs(deficit) > 1.0 or abs(global_target[k]) > 1.0:
            summaries.append({
                "hour": round(hour, 2),
                "target_kw": round(float(global_target[k]), 1),
                "aggregate_kw": round(float(aggregate[k]), 1),
                "deficit_kw": round(deficit, 1),
            })
    return summaries[:8]


def build_user_prompt(
    agent_id: int,
    iteration: int,
    global_target: np.ndarray,
    aggregate: np.ndarray,
    agent_share_kw: np.ndarray,
    config: VPPConfig = DEFAULT_CONFIG,
    previous_events: list[dict] | None = None,
    stale: bool = False,
) -> str:
    payload = {
        "agent_id": agent_id,
        "iteration": iteration,
        "grid_summary": {
            "evening_injection_need_kw": round(float(np.max(global_target)), 1),
            "morning_absorption_need_kw": round(float(np.min(global_target)), 1),
            "your_fair_share_evening_kw": round(float(np.max(agent_share_kw)), 1),
            "your_fair_share_morning_kw": round(float(np.min(agent_share_kw)), 1),
        },
        "deficit_next_6h": _deficit_summary(global_target, aggregate, config),
        "example_output": FEW_SHOT_EXAMPLE,
    }
    if previous_events:
        payload["your_previous_schedule"] = {"events": previous_events}
    if stale:
        payload["note"] = "Neighbor data may be stale due to network jitter."

    return "Plan your 24h battery calendar as JSON.\n" + json.dumps(payload, indent=2)


def calendar_to_events(calendar_probs: np.ndarray, config: VPPConfig) -> list[dict]:
    E = config.E
    dt = config.dwell_minutes / 60.0
    events: list[dict] = []
    prev_action = None
    for k in range(E):
        action_idx = int(np.argmax(calendar_probs[k]))
        action = config.action_names[action_idx]
        if action != prev_action:
            events.append({"start_hour": round(k * dt, 2), "action": action})
            prev_action = action
    return events
