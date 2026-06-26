"""Single LLM agent planning step with persona and extended guardrail support."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from credit_calculus.calendar import Calendar
from credit_calculus.config import VPPConfig, DEFAULT_CONFIG
from credit_calculus.guardrail_apply import apply_guardrail
from credit_calculus.guardrail_extended import apply_extended_guardrail
from credit_calculus.reward import interval_gradients
from llm_exp.client import LLMResponse, make_client
from llm_exp.config import DEFAULT_LLM_CONFIG, GuardrailMode, LLMConfig, Persona
from llm_exp.parser import parse_llm_calendar
from llm_exp.personas import system_prompt_for
from llm_exp.prompts import FeedbackContext, build_user_prompt, calendar_to_events


@dataclass
class PlanResult:
    raw_calendar: Calendar
    executed_calendar: Calendar
    parse_success: bool
    guardrail_modified: bool
    parse_error: str | None
    llm_response: LLMResponse | None
    guardrail_info: dict | None = None


class LLMAgent:
    def __init__(
        self,
        agent_id: int,
        config: VPPConfig = DEFAULT_CONFIG,
        llm_config: LLMConfig = DEFAULT_LLM_CONFIG,
        byzantine: bool = False,
        persona: Persona | None = None,
    ):
        self.agent_id = agent_id
        self.config = config
        self.llm_config = llm_config
        self.persona = persona or (Persona.BYZANTINE if byzantine else Persona.COOPERATIVE)
        self.byzantine = byzantine or self.persona == Persona.BYZANTINE
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
        guardrail_mode: GuardrailMode | None = None,
        stale: bool = False,
        feedback: FeedbackContext | None = None,
    ) -> PlanResult:
        prev_events = None
        if previous is not None:
            prev_events = calendar_to_events(previous.probs, self.config)

        system = system_prompt_for(self.persona)
        user = build_user_prompt(
            self.agent_id,
            iteration,
            global_target,
            aggregate,
            agent_share,
            self.config,
            previous_events=prev_events,
            stale=stale,
            feedback=feedback,
        )

        mode = guardrail_mode
        if mode is None:
            mode = GuardrailMode.FLOOR_ONLY if apply_guard else GuardrailMode.NONE

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
                guard_info: dict | None = None
                if mode != GuardrailMode.NONE:
                    if mode == GuardrailMode.FLOOR_ONLY:
                        executed, guard_info = apply_guardrail(
                            raw, neighbor_power, global_target, self.config
                        )
                    else:
                        executed, guard_info = apply_extended_guardrail(
                            raw,
                            neighbor_power,
                            global_target,
                            mode.value,
                            self.config,
                            agent_share=agent_share,
                        )
                    guard_modified = guard_info.get("modified", False)
                return PlanResult(
                    raw_calendar=raw,
                    executed_calendar=executed,
                    parse_success=True,
                    guardrail_modified=guard_modified,
                    parse_error=None,
                    llm_response=response,
                    guardrail_info=guard_info,
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
            guardrail_info=None,
        )

    def predicted_utility(
        self,
        calendar: Calendar,
        neighbor_power: np.ndarray,
        global_target: np.ndarray,
    ) -> float:
        """Numeric utility score for best-of-N selection."""
        G = interval_gradients(calendar, neighbor_power, global_target, self.config)
        dt = self.config.dwell_seconds
        return float(
            sum(dt * float(G[k] @ calendar.probs[k]) for k in range(self.config.E))
        )
