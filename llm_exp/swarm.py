"""LLM swarm simulator wrapping credit_calculus SwarmSimulator."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from credit_calculus.config import GraphonTopology, VPPConfig, DEFAULT_CONFIG
from credit_calculus.swarm import SwarmSimulator
from llm_exp.agent import LLMAgent, PlanResult
from llm_exp.config import LLMConfig, DEFAULT_LLM_CONFIG, PlanningCondition


@dataclass
class RoundMetrics:
    iteration: int
    tracking_error_pct: float
    parse_success_rate: float
    guardrail_modification_rate: float
    mean_llm_latency_ms: float
    total_prompt_tokens: int
    total_completion_tokens: int


@dataclass
class RunMetrics:
    condition: PlanningCondition
    iterations: int
    converged: bool
    final_tracking_pct: float
    errors: list[float] = field(default_factory=list)
    rounds: list[RoundMetrics] = field(default_factory=list)
    parse_success_rate: float = 0.0
    guardrail_modification_rate: float = 0.0
    mean_llm_latency_ms: float = 0.0


class LLMSwarmSimulator(SwarmSimulator):
    def __init__(
        self,
        n: int,
        condition: PlanningCondition = PlanningCondition.LLM_GUARDED,
        config: VPPConfig = DEFAULT_CONFIG,
        llm_config: LLMConfig = DEFAULT_LLM_CONFIG,
        topology: GraphonTopology = GraphonTopology.FLAT,
        rng: np.random.Generator | None = None,
        log_path: Path | None = None,
        stale_context: bool = False,
    ):
        super().__init__(n, config, topology, rng)
        self.condition = condition
        self.llm_config = llm_config
        self.log_path = log_path
        self.stale_context = stale_context
        self._agents = [
            LLMAgent(i, config, llm_config, byzantine=False) for i in range(n)
        ]
        self._published: list | None = None  # for schemer exp
        self._all_parse_ok: list[bool] = []
        self._all_guard_modified: list[bool] = []
        self._all_latencies: list[float] = []

    def mark_byzantine(self, fraction: float) -> None:
        super().mark_byzantine(fraction)
        for i in range(self.n):
            self._agents[i].byzantine = self.agent_types[i] == 1

    def _log(self, record: dict) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def step_llm(self, iteration: int = 0) -> RoundMetrics:
        if self.condition == PlanningCondition.BASELINE:
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
        apply_guard = self.condition == PlanningCondition.LLM_GUARDED

        round_parse: list[bool] = []
        round_guard: list[bool] = []
        round_lat: list[float] = []
        prompt_tok = 0
        comp_tok = 0
        new_calendars = []

        for i in range(self.n):
            if self.agent_types[i] == 1 and not self._agents[i].byzantine:
                new_calendars.append(self.calendars[i].copy())
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
                stale=self.stale_context,
            )
            new_calendars.append(result.executed_calendar)
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
                "iteration": iteration,
                "agent_id": i,
                "condition": self.condition.value,
                "parse_success": result.parse_success,
                "guardrail_modified": result.guardrail_modified,
                "parse_error": result.parse_error,
                "latency_ms": result.llm_response.latency_ms if result.llm_response else 0,
                "raw_preview": raw_preview,
            })

        self.calendars = new_calendars
        self._all_parse_ok.extend(round_parse)
        self._all_guard_modified.extend(round_guard)
        self._all_latencies.extend(round_lat)

        return RoundMetrics(
            iteration=iteration,
            tracking_error_pct=self.tracking_error_pct(),
            parse_success_rate=float(np.mean(round_parse)) if round_parse else 1.0,
            guardrail_modification_rate=float(np.mean(round_guard)) if round_guard else 0.0,
            mean_llm_latency_ms=float(np.mean(round_lat)) if round_lat else 0.0,
            total_prompt_tokens=prompt_tok,
            total_completion_tokens=comp_tok,
        )

    def run_until_converged(
        self,
        max_iter: int | None = None,
        condition: PlanningCondition | None = None,
    ) -> RunMetrics:
        if condition is not None:
            self.condition = condition
        max_iter = max_iter or self.config.max_iterations
        errors: list[float] = []
        rounds: list[RoundMetrics] = []

        for it in range(max_iter):
            prev = self.aggregate().copy()
            if self.condition == PlanningCondition.BASELINE:
                self.step_pga()
                rm = RoundMetrics(
                    iteration=it,
                    tracking_error_pct=self.tracking_error_pct(),
                    parse_success_rate=1.0,
                    guardrail_modification_rate=0.0,
                    mean_llm_latency_ms=0.0,
                    total_prompt_tokens=0,
                    total_completion_tokens=0,
                )
            else:
                rm = self.step_llm(iteration=it)
            errors.append(rm.tracking_error_pct)
            rounds.append(rm)
            delta = float(np.max(np.abs(self.aggregate() - prev)))
            if delta < self.config.convergence_tol and rm.tracking_error_pct < 5.0:
                return self._build_metrics(
                    it + 1, True, errors, rounds
                )

        return self._build_metrics(max_iter, False, errors, rounds)

    def _build_metrics(
        self,
        iterations: int,
        converged: bool,
        errors: list[float],
        rounds: list[RoundMetrics],
    ) -> RunMetrics:
        return RunMetrics(
            condition=self.condition,
            iterations=iterations,
            converged=converged,
            final_tracking_pct=errors[-1] if errors else 100.0,
            errors=errors,
            rounds=rounds,
            parse_success_rate=float(np.mean(self._all_parse_ok)) if self._all_parse_ok else 1.0,
            guardrail_modification_rate=float(np.mean(self._all_guard_modified))
            if self._all_guard_modified
            else 0.0,
            mean_llm_latency_ms=float(np.mean(self._all_latencies))
            if self._all_latencies
            else 0.0,
        )
