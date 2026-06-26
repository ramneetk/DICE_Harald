"""Single LLM agent planning step."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from credit_calculus.calendar import Calendar
from credit_calculus.config import VPPConfig, DEFAULT_CONFIG
from credit_calculus.guardrail_apply import apply_guardrail
from llm_exp.client import LLMResponse, make_client
from llm_exp.config import LLMConfig, DEFAULT_LLM_CONFIG
from llm_exp.parser import parse_llm_calendar
from llm_exp.prompts import (
    BYZANTINE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_user_prompt,
    calendar_to_events,
)


@dataclass
class PlanResult:
    raw_calendar: Calendar
    executed_calendar: Calendar
    parse_success: bool
    guardrail_modified: bool
    parse_error: str | None
    llm_response: LLMResponse | None


class LLMAgent:
    def __init__(
        self,
        agent_id: int,
        config: VPPConfig = DEFAULT_CONFIG,
        llm_config: LLMConfig = DEFAULT_LLM_CONFIG,
        byzantine: bool = False,
    ):
        self.agent_id = agent_id
        self.config = config
        self.llm_config = llm_config
        self.byzantine = byzantine
        self.client = make_client(llm_config)

    def plan(
        self,
        iteration: int,
        global_target: np.ndarray,
        aggregate: np.ndarray,
        agent_share: np.ndarray,
        neighbor_power: np.ndarray,
        previous: Calendar | None = None,
        apply_guard: bool = True,
        stale: bool = False,
    ) -> PlanResult:
        prev_events = None
        if previous is not None:
            prev_events = calendar_to_events(previous.probs, self.config)

        system = BYZANTINE_SYSTEM_PROMPT if self.byzantine else SYSTEM_PROMPT
        user = build_user_prompt(
            self.agent_id,
            iteration,
            global_target,
            aggregate,
            agent_share,
            self.config,
            previous_events=prev_events,
            stale=stale,
        )

        response = None
        text = ""
        for attempt in range(self.llm_config.max_retries + 1):
            response = self.client.complete(
                system,
                user if attempt == 0 else user + "\n\nYour last reply was invalid JSON. Fix it.",
                seed=self.llm_config.seed + self.agent_id + attempt,
            )
            text = response.text
            parsed = parse_llm_calendar(text, self.config)
            if parsed.success:
                raw = parsed.calendar
                executed = raw
                guard_modified = False
                if apply_guard:
                    executed, info = apply_guardrail(
                        raw, neighbor_power, global_target, self.config
                    )
                    guard_modified = info.get("modified", False)
                return PlanResult(
                    raw_calendar=raw,
                    executed_calendar=executed,
                    parse_success=True,
                    guardrail_modified=guard_modified,
                    parse_error=None,
                    llm_response=response,
                )

        parsed = parse_llm_calendar(text, self.config)
        idle = parsed.calendar
        return PlanResult(
            raw_calendar=idle,
            executed_calendar=idle,
            parse_success=False,
            guardrail_modified=False,
            parse_error=parsed.error,
            llm_response=response,
        )
