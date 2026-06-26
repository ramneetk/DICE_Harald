"""Steering swarm simulator with persona, guardrail mode, and credit feedback."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from credit_calculus.adversarial import deviation_penalty, realized_payout
from credit_calculus.config import GraphonTopology, VPPConfig, DEFAULT_CONFIG
from credit_calculus.credit import counterfactual_credit
from credit_calculus.graphon import neighborhood_indices
from credit_calculus.reward import interval_gradients
from llm_exp.agent import LLMAgent, PlanResult
from llm_exp.config import (
    DEFAULT_LLM_CONFIG,
    FeedbackMode,
    GuardrailMode,
    LLMConfig,
    Persona,
    PlanningCondition,
    SteeringRunConfig,
)
from llm_exp.prompts import FeedbackContext, gradient_hints_from_gradients
from llm_exp.swarm import LLMSwarmSimulator, RoundMetrics, RunMetrics


@dataclass
class SteeringRunMetrics:
    run_id: str
    persona: str
    guardrail_mode: str
    feedback_mode: str
    n: int
    iterations: int
    converged: bool
    final_tracking_pct: float
    errors: list[float] = field(default_factory=list)
    parse_success_rate: float = 0.0
    guardrail_modification_rate: float = 0.0
    mean_llm_latency_ms: float = 0.0


class SteeringSwarmSimulator(LLMSwarmSimulator):
    """LLM swarm with persona steering, extended guardrails, and credit feedback."""

    def __init__(
        self,
        n: int,
        run_config: SteeringRunConfig,
        config: VPPConfig = DEFAULT_CONFIG,
        llm_config: LLMConfig = DEFAULT_LLM_CONFIG,
        topology: GraphonTopology = GraphonTopology.FLAT,
        rng: np.random.Generator | None = None,
        log_path: Path | None = None,
        stale_context: bool = False,
    ):
        super().__init__(
            n,
            condition=PlanningCondition.LLM_GUARDED,
            config=config,
            llm_config=llm_config,
            topology=topology,
            rng=rng,
            log_path=log_path,
            stale_context=stale_context,
        )
        self.run_config = run_config
        self.persona = run_config.persona
        self.guardrail_mode = run_config.guardrail_mode
        self.feedback_mode = run_config.feedback_mode
        self._agents = [
            LLMAgent(i, config, llm_config, persona=run_config.persona)
            for i in range(n)
        ]
        self._feedback: list[FeedbackContext | None] = [None] * n
        self._published: list | None = None

    def _neighbor_indices(self, i: int) -> list[int]:
        return neighborhood_indices(self.graph, i, self.config.neighborhood_size)

    def _compute_feedback(self) -> None:
        """Compute per-agent feedback from current calendars for next round."""
        if self.feedback_mode == FeedbackMode.NONE:
            self._feedback = [None] * self.n
            return

        credits: list[float] = []
        penalties: list[float] = []
        payouts: list[float] = []

        for i in range(self.n):
            nbr = self.neighbor_power(i)
            cal = self.calendars[i]
            credit = counterfactual_credit(cal, nbr, self.global_target, self.config)
            credits.append(credit)

            pen = 0.0
            payout = credit
            if self._published is not None and self.feedback_mode == FeedbackMode.FULL:
                pub = self._published[i]
                pen = deviation_penalty(pub, cal, self.config)
                payout = realized_payout(credit, pen, self.config)
            penalties.append(pen)
            payouts.append(payout)

        ranked = np.argsort(np.argsort([-c for c in credits])) + 1

        feedback_list: list[FeedbackContext | None] = []
        for i in range(self.n):
            hints = None
            if self.feedback_mode in (
                FeedbackMode.GRADIENT_ONLY,
                FeedbackMode.CREDIT_GRADIENT,
                FeedbackMode.FULL,
            ):
                G = interval_gradients(
                    self.calendars[i],
                    self.neighbor_power(i),
                    self.global_target,
                    self.config,
                )
                hints = gradient_hints_from_gradients(G, self.config)

            feedback_list.append(
                FeedbackContext(
                    mode=self.feedback_mode,
                    credit=credits[i],
                    rank=int(ranked[i]),
                    swarm_size=self.n,
                    penalty=penalties[i] if self.feedback_mode == FeedbackMode.FULL else None,
                    payout=payouts[i] if self.feedback_mode == FeedbackMode.FULL else None,
                    gradient_hints=hints,
                )
            )
        self._feedback = feedback_list

    def step_steering(self, iteration: int = 0) -> RoundMetrics:
        if self.run_config.use_baseline:
            self.step_pga()
            return RoundMetrics(
                iteration=iteration,
                tracking_error_pct=self.tracking_error_pct(),
                parse_success_rate=1.0,
                guardrail_modification_rate=0.0,
                mean_llm_latency_ms=0.0,
                total_prompt_tokens=0,
                total_completion_tokens=0,
            )

        aggregate = self.aggregate()
        agent_share = self.global_target / max(self.n, 1)
        apply_guard = self.guardrail_mode != GuardrailMode.NONE

        round_parse: list[bool] = []
        round_guard: list[bool] = []
        round_lat: list[float] = []
        prompt_tok = 0
        comp_tok = 0
        new_calendars = []
        published_this_round: list = []

        for i in range(self.n):
            if self.agent_types[i] == 1 and self.persona != Persona.BYZANTINE:
                new_calendars.append(self.calendars[i].copy())
                published_this_round.append(self.calendars[i].copy())
                continue

            nbr = self.neighbor_power(i)
            result: PlanResult = self._agents[i].plan(
                iteration=iteration,
                global_target=self.global_target,
                aggregate=aggregate,
                agent_share=agent_share,
                neighbor_power=nbr,
                previous=self.calendars[i],
                apply_guard=apply_guard,
                guardrail_mode=self.guardrail_mode,
                stale=self.stale_context,
                feedback=self._feedback[i],
            )
            new_calendars.append(result.executed_calendar)
            published_this_round.append(result.raw_calendar)
            round_parse.append(result.parse_success)
            round_guard.append(result.guardrail_modified)
            if result.llm_response:
                round_lat.append(result.llm_response.latency_ms)
                prompt_tok += result.llm_response.prompt_tokens
                comp_tok += result.llm_response.completion_tokens

            raw_preview = ""
            if result.llm_response and result.llm_response.text:
                raw_preview = result.llm_response.text[:800]
            self._log({
                "run_id": self.run_config.run_id,
                "iteration": iteration,
                "agent_id": i,
                "persona": self.persona.value,
                "guardrail_mode": self.guardrail_mode.value,
                "feedback_mode": self.feedback_mode.value,
                "parse_success": result.parse_success,
                "guardrail_modified": result.guardrail_modified,
                "parse_error": result.parse_error,
                "latency_ms": result.llm_response.latency_ms if result.llm_response else 0,
                "raw_preview": raw_preview,
                "credit": self._feedback[i].credit if self._feedback[i] else None,
            })

        self._published = published_this_round
        self.calendars = new_calendars
        self._all_parse_ok.extend(round_parse)
        self._all_guard_modified.extend(round_guard)
        self._all_latencies.extend(round_lat)

        self._compute_feedback()

        return RoundMetrics(
            iteration=iteration,
            tracking_error_pct=self.tracking_error_pct(),
            parse_success_rate=float(np.mean(round_parse)) if round_parse else 1.0,
            guardrail_modification_rate=float(np.mean(round_guard)) if round_guard else 0.0,
            mean_llm_latency_ms=float(np.mean(round_lat)) if round_lat else 0.0,
            total_prompt_tokens=prompt_tok,
            total_completion_tokens=comp_tok,
        )

    def run_steering(self, max_iter: int | None = None) -> SteeringRunMetrics:
        max_iter = max_iter or self.run_config.rounds
        errors: list[float] = []

        for it in range(max_iter):
            prev = self.aggregate().copy()
            if self.run_config.use_baseline:
                self.step_pga()
                errors.append(self.tracking_error_pct())
            else:
                self.step_steering(iteration=it)
                errors.append(self.tracking_error_pct())

            delta = float(np.max(np.abs(self.aggregate() - prev)))
            if (
                delta < self.config.convergence_tol
                and errors[-1] < 5.0
            ):
                return self._build_steering_metrics(it + 1, True, errors)

        return self._build_steering_metrics(max_iter, False, errors)

    def _build_steering_metrics(
        self,
        iterations: int,
        converged: bool,
        errors: list[float],
    ) -> SteeringRunMetrics:
        return SteeringRunMetrics(
            run_id=self.run_config.run_id,
            persona=self.persona.value,
            guardrail_mode=self.guardrail_mode.value,
            feedback_mode=self.feedback_mode.value,
            n=self.n,
            iterations=iterations,
            converged=converged,
            final_tracking_pct=errors[-1] if errors else 100.0,
            errors=errors,
            parse_success_rate=float(np.mean(self._all_parse_ok))
            if self._all_parse_ok
            else 1.0,
            guardrail_modification_rate=float(np.mean(self._all_guard_modified))
            if self._all_guard_modified
            else 0.0,
            mean_llm_latency_ms=float(np.mean(self._all_latencies))
            if self._all_latencies
            else 0.0,
        )


def default_steering_matrix() -> list[SteeringRunConfig]:
    """R0–R6 experiment matrix from the research plan."""
    return [
        SteeringRunConfig(
            run_id="R0",
            persona=Persona.COOPERATIVE,
            guardrail_mode=GuardrailMode.NONE,
            feedback_mode=FeedbackMode.NONE,
            n=10,
            rounds=8,
            use_baseline=True,
        ),
        SteeringRunConfig(
            run_id="R1",
            persona=Persona.COOPERATIVE,
            guardrail_mode=GuardrailMode.FLOOR_ONLY,
            feedback_mode=FeedbackMode.NONE,
            n=10,
            rounds=8,
        ),
        SteeringRunConfig(
            run_id="R2",
            persona=Persona.AGGRESSIVE,
            guardrail_mode=GuardrailMode.FLOOR_ONLY,
            feedback_mode=FeedbackMode.NONE,
            n=10,
            rounds=8,
        ),
        SteeringRunConfig(
            run_id="R3",
            persona=Persona.AGGRESSIVE,
            guardrail_mode=GuardrailMode.ATTRACTOR_SEEK,
            feedback_mode=FeedbackMode.GRADIENT_ONLY,
            n=10,
            rounds=8,
        ),
        SteeringRunConfig(
            run_id="R4",
            persona=Persona.MISSION_CRITICAL,
            guardrail_mode=GuardrailMode.HYBRID_BAND,
            feedback_mode=FeedbackMode.CREDIT_GRADIENT,
            n=10,
            rounds=8,
        ),
        SteeringRunConfig(
            run_id="R5",
            persona=Persona.AGGRESSIVE,
            guardrail_mode=GuardrailMode.ATTRACTOR_SEEK,
            feedback_mode=FeedbackMode.FULL,
            n=10,
            rounds=8,
        ),
        SteeringRunConfig(
            run_id="R6",
            persona=Persona.AGGRESSIVE,
            guardrail_mode=GuardrailMode.ATTRACTOR_SEEK,
            feedback_mode=FeedbackMode.FULL,
            n=25,
            rounds=8,
        ),
    ]
