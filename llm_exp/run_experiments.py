#!/usr/bin/env python3
"""
LLM agent experiments — Qwen3.5-0.8B via vLLM.

Mirrors the numeric Credit Calculus experiments under three conditions:
  baseline | llm_raw | llm_guarded
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from credit_calculus.adversarial import (
    byzantine_eviction_rounds,
    deviation_penalty,
    realized_payout,
    schemer_executed_calendar,
    schemer_published_calendar,
)
from credit_calculus.config import DEFAULT_CONFIG, QUICK_CONFIG, VPPConfig
from credit_calculus.guardrail import benchmark_guardrail_ms
from llm_exp.client import server_reachable
from llm_exp.config import DEFAULT_LLM_CONFIG, LLMConfig, PlanningCondition
from llm_exp.plots import (
    ensure_output_dir,
    plot_convergence,
    plot_guardrail_effect,
    plot_latency,
    plot_parse_failures,
    plot_tracking_comparison,
    save_summary_csv,
)
from llm_exp.swarm import LLMSwarmSimulator


def parse_conditions(raw: str) -> list[PlanningCondition]:
    mapping = {c.value: c for c in PlanningCondition}
    return [mapping[s.strip()] for s in raw.split(",") if s.strip()]


def run_tracking_experiment(
    n: int,
    config: VPPConfig,
    llm_config: LLMConfig,
    conditions: list[PlanningCondition],
    max_iter: int,
    log_dir: Path,
) -> dict[str, dict]:
    results = {}
    for cond in conditions:
        log_path = log_dir / f"n{n}_{cond.value}.jsonl"
        sim = LLMSwarmSimulator(
            n,
            condition=cond,
            config=config,
            llm_config=llm_config,
            log_path=log_path,
        )
        metrics = sim.run_until_converged(max_iter=max_iter, condition=cond)
        results[cond.value] = {
            "final_tracking_pct": metrics.final_tracking_pct,
            "iterations": metrics.iterations,
            "converged": metrics.converged,
            "errors": metrics.errors,
            "parse_success_rate": metrics.parse_success_rate,
            "guardrail_modification_rate": metrics.guardrail_modification_rate,
            "mean_llm_latency_ms": metrics.mean_llm_latency_ms,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Credit Calculus experiments")
    parser.add_argument("--output", type=Path, default=Path("llm_exp/results"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM (no vLLM)")
    parser.add_argument(
        "--conditions",
        default="baseline,llm_raw,llm_guarded",
        help="Comma-separated: baseline,llm_raw,llm_guarded",
    )
    parser.add_argument("--n", type=int, nargs="*", default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--model",
        default=None,
        help="Override vLLM model id (auto-detect from /v1/models if omitted)",
    )
    args = parser.parse_args()

    model_id = args.model
    if model_id is None and not args.mock:
        try:
            import urllib.request
            import json as _json

            url = args.vllm_url.rstrip("/") + "/models"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = _json.loads(resp.read())
            models = [m["id"] for m in data.get("data", [])]
            if models:
                model_id = models[0]
                print(f"Auto-detected vLLM model: {model_id}", flush=True)
        except Exception:
            model_id = DEFAULT_LLM_CONFIG.model

    vpp_config = QUICK_CONFIG if args.quick else DEFAULT_CONFIG
    llm_config = replace(
        DEFAULT_LLM_CONFIG,
        base_url=args.vllm_url,
        use_mock=args.mock,
        model=model_id or DEFAULT_LLM_CONFIG.model,
    )

    if not llm_config.use_mock and not server_reachable(llm_config):
        print("vLLM not reachable at", llm_config.base_url)
        print("Start server: llm_exp/scripts/serve_qwen.sh")
        print("Or use --mock for offline smoke test.")
        sys.exit(1)

    if args.n:
        n_values = args.n
    elif args.quick:
        n_values = [10, 25]
    else:
        n_values = [10, 25, 50, 100]

    max_iter = args.rounds or (5 if args.quick else 15)
    conditions = parse_conditions(args.conditions)

    out_dir = ensure_output_dir(args.output)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    print("LLM Credit Calculus experiments", flush=True)
    print(f"  conditions: {[c.value for c in conditions]}", flush=True)
    print(f"  n={n_values}, rounds={max_iter}, mock={llm_config.use_mock}", flush=True)
    print(f"  model: {llm_config.model}", flush=True)
    print(f"  output: {out_dir.resolve()}", flush=True)

    t0 = time.time()
    summary_rows = []
    tracking_by_cond: dict[str, list[float]] = {c.value: [] for c in conditions}
    parse_rates: list[float] = []
    llm_latencies: list[float] = []
    guard_ms = benchmark_guardrail_ms(vpp_config.E, vpp_config.num_actions, vpp_config)

    convergence_sample: dict | None = None

    for n in n_values:
        print(f"\n[n={n}] Running conditions...", flush=True)
        t_n = time.time()
        per_n = run_tracking_experiment(
            n, vpp_config, llm_config, conditions, max_iter, log_dir
        )
        wall_s = time.time() - t_n
        for cond_name, data in per_n.items():
            tracking_by_cond[cond_name].append(data["final_tracking_pct"])
            summary_rows.append({
                "n": n,
                "condition": cond_name,
                "model": llm_config.model,
                "tracking_pct": round(data["final_tracking_pct"], 2),
                "iterations": data["iterations"],
                "converged": data["converged"],
                "parse_success_rate": round(data["parse_success_rate"], 3),
                "guardrail_mod_rate": round(data["guardrail_modification_rate"], 3),
                "mean_llm_ms": round(data["mean_llm_latency_ms"], 2),
                "wall_time_s": round(wall_s, 1),
            })
            if cond_name == "llm_guarded":
                parse_rates.append(data["parse_success_rate"])
                llm_latencies.append(data["mean_llm_latency_ms"])

        if convergence_sample is None and "baseline" in per_n:
            convergence_sample = {
                "baseline_errors": per_n.get("baseline", {}).get("errors", []),
                "llm_raw_errors": per_n.get("llm_raw", {}).get("errors", []),
                "llm_guarded_errors": per_n.get("llm_guarded", {}).get("errors", []),
            }

    plot_data = {
        "n_values": n_values,
        "baseline": tracking_by_cond.get("baseline", []),
        "llm_raw": tracking_by_cond.get("llm_raw", []),
        "llm_guarded": tracking_by_cond.get("llm_guarded", []),
        "parse_success_rate": parse_rates,
        "llm_latency_ms": llm_latencies,
        "guardrail_ms": [guard_ms] * len(n_values),
    }
    plot_tracking_comparison(plot_data, out_dir)
    plot_guardrail_effect(plot_data, out_dir)
    plot_parse_failures(plot_data, out_dir)
    plot_latency(plot_data, out_dir)
    if convergence_sample:
        plot_convergence(convergence_sample, out_dir)

    pub = schemer_published_calendar(vpp_config)
    exe = schemer_executed_calendar(vpp_config)
    pen = deviation_penalty(pub, exe, vpp_config)
    schemer_payout = realized_payout(5.0, pen, vpp_config)

    save_summary_csv(summary_rows, out_dir)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Figures: {out_dir}/*.png")
    print(f"Table:   {out_dir}/table_llm_summary.csv")
    print(f"Logs:    {log_dir}/")
    print(f"Schemer non-Markovian payout (Exp 4): {schemer_payout}")
    for row in summary_rows[:6]:
        print(row)


if __name__ == "__main__":
    main()
