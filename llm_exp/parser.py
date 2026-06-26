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

ACTION_ALIASES: dict[str, str] = {
    "idle": "Idle",
    "charge": "ChargeStandard",
    "chargestandard": "ChargeStandard",
    "charge_standard": "ChargeStandard",
    "chargefast": "ChargeFast",
    "charge_fast": "ChargeFast",
    "fastcharge": "ChargeFast",
    "discharge": "DischargeV2G",
    "dischargev2g": "DischargeV2G",
    "discharge_v2g": "DischargeV2G",
    "v2g": "DischargeV2G",
    "inject": "DischargeV2G",
}


@dataclass
class ParseResult:
    calendar: Calendar
    success: bool
    error: str | None = None


def _strip_thinking(text: str) -> str:
    """Remove Qwen-style reasoning blocks before JSON extraction."""
    text = text.strip()
    bt = chr(96)
    think_open = bt + "think" + bt
    think_close = bt + "/think" + bt
    text = re.sub(
        re.escape(think_open) + r"[\s\S]*?" + re.escape(think_close),
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"<\|think\|>[\s\S]*?<\|/think\|>", "", text, flags=re.IGNORECASE)
    if re.search(r"(?i)^thinking process:", text):
        brace = text.find("{")
        if brace != -1:
            text = text[brace:]
    return text.strip()


def _normalize_action(raw: str) -> str | None:
    if raw in VALID_ACTIONS:
        return raw
    key = re.sub(r"[\s_-]+", "", raw.strip().lower())
    return ACTION_ALIASES.get(key)


def _parse_event_objects(text: str) -> list[dict]:
    """Salvage complete event dicts from full or truncated JSON."""
    events: list[dict] = []
    for match in re.finditer(
        r'\{\s*"start_hour"\s*:\s*([0-9.]+)\s*,\s*"action"\s*:\s*"([^"]+)"\s*\}',
        text,
    ):
        action = _normalize_action(match.group(2))
        if action is None:
            continue
        events.append({"start_hour": float(match.group(1)), "action": action})
    return events


def _extract_json(text: str) -> dict:
    text = _strip_thinking(text)
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start == -1:
        events = _parse_event_objects(text)
        if events:
            return {"events": events}
        raise ValueError("no JSON object found")

    snippet = text[start:]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        end = snippet.rfind("}")
        if end != -1:
            try:
                return json.loads(snippet[: end + 1])
            except json.JSONDecodeError:
                pass
        events = _parse_event_objects(snippet)
        if events:
            return {"events": events}
        raise


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
        action = _normalize_action(str(ev.get("action", "Idle")))
        if action is None:
            raise ValueError(f"invalid action: {ev.get('action')}")
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
    if not text or not text.strip():
        return ParseResult(
            calendar=Calendar.idle(config),
            success=False,
            error="empty response",
        )
    try:
        data = _extract_json(text)
        events = data.get("events")
        if events is None and isinstance(data, list):
            events = data
        if not isinstance(events, list):
            raise ValueError("events must be a list")
        if not events:
            raise ValueError("events list is empty")
        cal = events_to_calendar(events, config)
        return ParseResult(calendar=cal, success=True)
    except Exception as exc:
        return ParseResult(
            calendar=Calendar.idle(config),
            success=False,
            error=str(exc),
        )
