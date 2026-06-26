"""Prompt templates for VPP calendar planning."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from credit_calculus.config import VPPConfig, DEFAULT_CONFIG
from llm_exp.config import FeedbackMode
from llm_exp.personas import BYZANTINE_SYSTEM_PROMPT, SYSTEM_PROMPT

__all__ = [
    "SYSTEM_PROMPT",
    "BYZANTINE_SYSTEM_PROMPT",
    "FEW_SHOT_EXAMPLE",
    "FeedbackContext",
    "build_user_prompt",
    "calendar_to_events",
    "gradient_hints_from_gradients",
]


@dataclass
class FeedbackContext:
    """Prior-round feedback injected into user prompts."""

    mode: FeedbackMode = FeedbackMode.NONE
    credit: float | None = None
    rank: int | None = None
    swarm_size: int | None = None
    penalty: float | None = None
    payout: float | None = None
    gradient_hints: list[dict] | None = None


FEW_SHOT_EXAMPLE = {
    "events": [
        {"start_hour": 0.0, "action": "Idle"},
        {"start_hour": 3.0, "action": "ChargeFast"},
        {"start_hour": 5.0, "action": "Idle"},
        {"start_hour": 17.25, "action": "DischargeV2G"},
        {"start_hour": 19.0, "action": "Idle"},
    ]
}


def gradient_hints_from_gradients(
    gradients: np.ndarray,
    config: VPPConfig = DEFAULT_CONFIG,
    max_hints: int = 4,
) -> list[dict]:
    """Top action per interval by gradient magnitude (for prompt hints)."""
    dt = config.dwell_minutes / 60.0
    hints: list[dict] = []
    for k in range(min(config.E, max_hints * 3)):
        g = gradients[k]
        if float(np.linalg.norm(g)) < 1e-9:
            continue
        action_idx = int(np.argmax(g))
        hour = round((k + 0.5) * dt, 2)
        hints.append({
            "hour": hour,
            "action": config.action_names[action_idx],
        })
        if len(hints) >= max_hints:
            break
    return hints


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
    return summaries[:6]


def _feedback_lines(feedback: FeedbackContext | None) -> list[str]:
    if feedback is None or feedback.mode == FeedbackMode.NONE:
        return []

    lines: list[str] = []
    mode = feedback.mode

    if mode in (
        FeedbackMode.CREDIT_ONLY,
        FeedbackMode.CREDIT_GRADIENT,
        FeedbackMode.FULL,
    ):
        if feedback.credit is not None:
            rank_str = ""
            if feedback.rank is not None and feedback.swarm_size is not None:
                rank_str = f" Rank: {feedback.rank}/{feedback.swarm_size}."
            lines.append(
                f"Your last-round Shapley credit: {feedback.credit:.2f}.{rank_str}"
            )

    if mode in (FeedbackMode.GRADIENT_ONLY, FeedbackMode.CREDIT_GRADIENT, FeedbackMode.FULL):
        if feedback.gradient_hints:
            parts = [
                f"{h['hour']}h→{h['action']}" for h in feedback.gradient_hints[:4]
            ]
            lines.append(
                "Recommended interval actions (gradient hints): " + ", ".join(parts) + "."
            )

    if mode == FeedbackMode.FULL:
        if feedback.penalty is not None:
            lines.append(f"Deviation penalty risk: {feedback.penalty:.2f}.")
        if feedback.payout is not None:
            lines.append(f"Non-Markovian payout: {feedback.payout:.2f}.")

    return lines


def build_user_prompt(
    agent_id: int,
    iteration: int,
    global_target: np.ndarray,
    aggregate: np.ndarray,
    agent_share_kw: np.ndarray,
    config: VPPConfig = DEFAULT_CONFIG,
    previous_events: list[dict] | None = None,
    stale: bool = False,
    feedback: FeedbackContext | None = None,
) -> str:
    deficits = _deficit_summary(global_target, aggregate, config)
    lines = [
        f"Agent {agent_id}, planning round {iteration}.",
        (
            "Grid: evening injection need "
            f"{round(float(np.max(global_target)), 1)} kW total "
            f"(your fair share {round(float(np.max(agent_share_kw)), 1)} kW); "
            "morning absorption need "
            f"{round(float(np.min(global_target)), 1)} kW total "
            f"(your fair share {round(float(np.min(agent_share_kw)), 1)} kW)."
        ),
    ]
    if deficits:
        parts = [
            f"{d['hour']}h target {d['target_kw']} aggregate {d['aggregate_kw']}"
            for d in deficits[:4]
        ]
        lines.append("Near-term deficits: " + "; ".join(parts) + ".")
    lines.extend(_feedback_lines(feedback))
    if previous_events:
        lines.append(
            "Previous schedule: "
            + json.dumps({"events": previous_events}, separators=(",", ":"))
        )
    if stale:
        lines.append("Note: neighbor data may be stale.")
    lines.append(
        "Output JSON only matching this example: "
        + json.dumps(FEW_SHOT_EXAMPLE, separators=(",", ":"))
    )
    return "\n".join(lines)


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
