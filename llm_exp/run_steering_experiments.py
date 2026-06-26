#!/usr/bin/env python3
"""
Steering experiments — push LLM agents outside the safety bubble.

Runs the R0–R6 matrix: persona × guardrail mode × credit feedback.
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

from credit_calculus.config import DEFAULT_CONFIG, QUICK_CONFIG, VPPConfig
from llm_exp.client import server_reachable
from llm_exp.config import (
    DEFAULT_LLM_CONFIG,
    FeedbackMode,
    GuardrailMode,
    LLMConfig,
    Persona,
    SteeringRunConfig,
)
from llm_exp.plots import (
    ensure_output_dir,
    plot_feedback_ablation,
    plot_guardrail_pareto,
    plot_persona_sweep,
    plot_steering_summary,
    save_summary_csv,
)
from llm_exp.steering_swarm import SteeringSwarmSimulator, default_steering_matrix


def run_steering_experiment(
    run_config: SteeringRunConfig,
    vpp_config: VPPConfig,
    llm_config: LLMConfig,
    log_dir: Path,
) -> dict:
    log_path = log_dir / f"{run_config.run_id}_n{run_config.n}.jsonl"
    sim = SteeringSwarmSimulator(
        run_config.n,
        run_config=run_config,
        config=vpp_config,
        llm_config=llm_config,
        log_path=log_path,
    )
    t0 = time.time()
    metrics = sim.run_steering(max_iter=run_config.rounds)
    wall_s = time.time() - t0
    return {
        "run_id": run_config.run_id,
        "persona": metrics.persona,
        "guardrail_mode": metrics.guardrail_mode,
        "feedback_mode": metrics.feedback_mode,
        "n": metrics.n,
        "tracking_pct": round(metrics.final_tracking_pct, 2),
        "iterations": metrics.iterations,
        "converged": metrics.converged,
        "parse_success_rate": round(metrics.parse_success_rate, 3),
        "guardrail_mod_rate": round(metrics.guardrail_modification_rate, 3),
        "mean_llm_ms": round(metrics.mean_llm_latency_ms, 2),
        "wall_time_s": round(wall_s, 1),
        "use_baseline": run_config.use_baseline,
        "model": llm_config.model,
        "errors": metrics.errors,
    }


def parse_run_ids(raw: str, matrix: list[SteeringRunConfig]) -> list[SteeringRunConfig]:
    if raw.strip().lower() == "all":
        return matrix
    wanted = {s.strip().upper() for s in raw.split(",") if s.strip()}
    return [r for r in matrix if r.run_id in wanted]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Steering LLM agents outside the safety bubble (R0–R6)"
    )
    parser.add_argument("--output", type=Path, default=Path("llm_exp/results_steering"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM (no vLLM)")
    parser.add_argument(
        "--runs",
        default="all",
        help="Comma-separated run IDs (R0,R1,...) or 'all'",
    )
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    model_id = args.model
    if model_id is None and not args.mock:
        try:
            import json as _json
            import urllib.request

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

    matrix = default_steering_matrix()
    if args.rounds is not None:
        matrix = [
            replace(r, rounds=args.rounds) for r in matrix
        ]
    runs = parse_run_ids(args.runs, matrix)

    out_dir = ensure_output_dir(args.output)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    print("Steering experiments (safety bubble research)", flush=True)
    print(f"  runs: {[r.run_id for r in runs]}", flush=True)
    print(f"  mock={llm_config.use_mock}, model={llm_config.model}", flush=True)
    print(f"  output: {out_dir.resolve()}", flush=True)

    t0 = time.time()
    summary_rows: list[dict] = []
    plot_rows: list[dict] = []

    for run_config in runs:
        print(f"\n[{run_config.run_id}] persona={run_config.persona.value} "
              f"guardrail={run_config.guardrail_mode.value} "
              f"feedback={run_config.feedback_mode.value} n={run_config.n}", flush=True)
        result = run_steering_experiment(run_config, vpp_config, llm_config, log_dir)
        row = {k: v for k, v in result.items() if k != "errors"}
        summary_rows.append(row)
        plot_rows.append(row)
        print(f"  tracking={result['tracking_pct']}% converged={result['converged']}", flush=True)

    save_summary_csv(summary_rows, out_dir, filename="table_steering_summary.csv")
    plot_steering_summary(plot_rows, out_dir)
    plot_persona_sweep(plot_rows, out_dir)
    plot_guardrail_pareto(plot_rows, out_dir)
    plot_feedback_ablation(plot_rows, out_dir)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Table:   {out_dir}/table_steering_summary.csv")
    print(f"Figures: {out_dir}/fig_steering_*.png")
    print(f"Logs:    {log_dir}/")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
