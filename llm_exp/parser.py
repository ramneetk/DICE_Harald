"""Parse LLM JSON event-list output into Credit Calculus calendars."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

from credit_calculus.calendar import Calendar
from credit_calculus.config import VPPConfig, DEFAULT_CONFIG

VALID_ACTIONS = frozenset(
    {"Idle", "ChargeStandard", "ChargeFast", "DischargeV2G"}
)


@dataclass
class ParseResult:
    calendar: Calendar
    success: bool
    error: str | None = None


def _extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    return json.loads(text[start : end + 1])


def events_to_calendar(
    events: list[dict],
    config: VPPConfig = DEFAULT_CONFIG,
) -> Calendar:
    """Convert sorted event list to piecewise-constant calendar."""
    E = config.E
    dt_hours = config.dwell_minutes / 60.0
    action_to_idx = {name: i for i, name in enumerate(config.action_names)}

    if not events:
        return Calendar.idle(config)

    parsed: list[tuple[float, int]] = []
    for ev in events:
        action = str(ev.get("action", "Idle"))
        if action not in VALID_ACTIONS:
            raise ValueError(f"invalid action: {action}")
        hour = float(ev.get("start_hour", ev.get("hour", 0.0)))
        parsed.append((hour, action_to_idx[action]))

    parsed.sort(key=lambda x: x[0])

    probs = np.zeros((E, config.num_actions))
    for k in range(E):
        hour = (k + 0.5) * dt_hours
        action_idx = parsed[0][1]
        for start_hour, idx in parsed:
            if start_hour <= hour:
                action_idx = idx
            else:
                break
        probs[k, action_idx] = 1.0

    return Calendar(probs, config)


def parse_llm_calendar(
    text: str,
    config: VPPConfig = DEFAULT_CONFIG,
) -> ParseResult:
    try:
        data = _extract_json(text)
        events = data.get("events", data if isinstance(data, list) else [])
        if not isinstance(events, list):
            raise ValueError("events must be a list")
        cal = events_to_calendar(events, config)
        return ParseResult(calendar=cal, success=True)
    except Exception as exc:
        return ParseResult(
            calendar=Calendar.idle(config),
            success=False,
            error=str(exc),
        )
