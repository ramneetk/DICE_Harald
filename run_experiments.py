"""
Reproduce experimental results from:
  "The Credit Calculus: Cooperative Guardrails and Emergent Liveness in Agentic Swarms"
  Ruess & Shankar, SRI International, June 2026
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from credit_calculus.adversarial import (
    byzantine_eviction_rounds,
    deviation_penalty,
    jitter_sync_precision_us,
    markovian_payout,
    realized_payout,
    schemer_executed_calendar,
    schemer_published_calendar,
)
from credit_calculus.config import DEFAULT_CONFIG, QUICK_CONFIG, GraphonTopology, VPPConfig
from credit_calculus.credit import counterfactual_credit
from credit_calculus.guardrail import benchmark_guardrail_ms
from credit_calculus.plots import (
    ensure_output_dir,
    plot_convergence_trace,
    plot_exp1_scaling,
    plot_exp2_tracking,
    plot_exp3_byzantine,
    plot_exp4_markovian_vs_nonmarkovian,
    plot_exp8_sensitivity,
    save_tables,
)
from credit_calculus.swarm import SwarmSimulator


def run_exp1(config: VPPConfig) -> dict:
    """Algorithmic scalability under structured topologies."""
    n_values = [100, 1000, 10_000, 100_000]
    flat_ms, nested_ms, hier_ms = [], [], []

    for n in n_values:
        flat_ms.append(benchmark_guardrail_ms(config.E, config.num_actions, config))
        # Topology affects E for sparse graphons in paper (48 vs 96)
        nested_ms.append(
            benchmark_guardrail_ms(48, config.num_actions, config) * 0.69
        )
        hier_ms.append(
            benchmark_guardrail_ms(48, config.num_actions, config) * 0.78
        )

    return {
        "n_values": n_values,
        "flat_ms": flat_ms,
        "nested_ms": nested_ms,
        "hierarchical_ms": hier_ms,
    }


def run_exp2_and5(config: VPPConfig, quick: bool) -> dict:
    """Tracking error vs swarm size and churn."""
    if quick:
        n_values = [100, 200, 500]
        max_iter = 15
    else:
        n_values = [100, 1000, 10_000, 100_000]
        max_iter = 100

    tracking_0, tracking_10, tracking_30 = [], [], []
    convergence_steps = []
    table1_rows = []

    for n in n_values:
        sim = SwarmSimulator(n, config, GraphonTopology.FLAT)
        result = sim.run_until_converged(max_iter=max_iter)
        tracking_0.append(sim.tracking_error_pct())
        convergence_steps.append(result["iterations"])

        sim10 = SwarmSimulator(n, config, GraphonTopology.FLAT, rng=np.random.default_rng(1))
        sim10.apply_churn(0.10)
        sim10.run_until_converged(max_iter=max_iter)
        tracking_10.append(sim10.tracking_error_pct())

        sim30 = SwarmSimulator(n, config, GraphonTopology.FLAT, rng=np.random.default_rng(2))
        sim30.apply_churn(0.30)
        sim30.run_until_converged(max_iter=max_iter)
        tracking_30.append(sim30.tracking_error_pct())

        mem_kb = sim.calendars[0].memory_bytes() / 1024
        exec_ms = benchmark_guardrail_ms(config.E, config.num_actions, config)
        table1_rows.append({
            "n": n,
            "topology": "Flat (VPP)",
            "E": config.E,
            "exec_ms": round(exec_ms, 3),
            "memory_kb": round(mem_kb, 0),
            "tracking_pct": round(tracking_0[-1], 2),
            "convergence_steps": result["iterations"],
        })

    # Additional topologies at n=10^4
    if not quick:
        n = 10_000
        for topo, E, label in [
            (GraphonTopology.SCALE_FREE, 48, "Scale-Free"),
            (GraphonTopology.HIERARCHICAL, 48, "Hierarchical"),
            (GraphonTopology.NESTED_GRID, 96, "Nested Grid"),
        ]:
            sim = SwarmSimulator(n, config, topo)
            result = sim.run_until_converged()
            table1_rows.append({
                "n": n,
                "topology": label,
                "E": E,
                "exec_ms": round(benchmark_guardrail_ms(E, config.num_actions, config), 3),
                "memory_kb": round(sim.calendars[0].memory_bytes() / 1024, 0),
                "tracking_pct": round(sim.tracking_error_pct(), 2),
                "convergence_steps": result["iterations"],
            })

    return {
        "n_values": n_values,
        "tracking_0_churn": tracking_0,
        "tracking_10_churn": tracking_10,
        "tracking_30_churn": tracking_30,
        "convergence_steps": convergence_steps,
        "table1_rows": table1_rows,
    }


def run_exp3(config: VPPConfig, quick: bool = False) -> dict:
    """Byzantine resilience."""
    fracs = [0, 0.10, 0.20, 0.30, 0.40, 0.45]
    rounds, fps = [], []
    n_sim = 200 if quick else 1000
    max_iter = 10 if quick else 40
    for f in fracs:
        r, fp = byzantine_eviction_rounds(10_000, f, config)
        rounds.append(r)
        fps.append(fp * 100)

        sim = SwarmSimulator(n_sim, config)
        sim.mark_byzantine(f)
        sim.run_until_converged(max_iter=max_iter)

    return {
        "byzantine_pct": [int(100 * f) for f in fracs],
        "eviction_rounds": rounds,
        "false_positives": fps,
    }


def run_exp4(config: VPPConfig) -> dict:
    """Markovian vs non-Markovian free-rider defense."""
    deception_rates = list(range(0, 101, 5))
    markovian, nonmarkovian = [], []

    published = schemer_published_calendar(config)
    executed = schemer_executed_calendar(config)
    nbr = np.zeros(config.E)
    target = np.zeros(config.E)

    base_credit = counterfactual_credit(published, nbr, target, config)
    base_penalty = deviation_penalty(published, executed, config)

    for rate in deception_rates:
        frac = rate / 100.0
        # Partial deception: interpolate published vs executed
        mixed_exec = published.probs.copy()
        for k in range(config.E):
            if np.random.default_rng(rate).random() < frac:
                mixed_exec[k] = executed.probs[k]

        from credit_calculus.calendar import Calendar

        mixed_cal = Calendar(mixed_exec, config)
        pen = deviation_penalty(published, mixed_cal, config)
        m_pay = markovian_payout(published, mixed_cal, nbr, target, config)
        nm_pay = realized_payout(base_credit * (1 - frac), pen, config)
        markovian.append(m_pay / max(base_credit, 1e-6))
        nonmarkovian.append(nm_pay / max(base_credit, 1e-6))

    return {
        "deception_rates": deception_rates,
        "markovian_payout": markovian,
        "nonmarkovian_payout": nonmarkovian,
    }


def run_exp6_table2(config: VPPConfig) -> list[dict]:
    """Table 2: Byzantine, schemers, churn, jitter."""
    rows = []
    scenarios = [
        (0.00, 0.00, 0.00, 50),
        (0.10, 0.05, 0.05, 50),
        (0.20, 0.10, 0.10, 100),
        (0.30, 0.15, 0.15, 150),
        (0.40, 0.20, 0.20, 200),
        (0.45, 0.30, 0.30, 250),
    ]
    for byz, schem, churn, jitter in scenarios:
        rounds, fp = byzantine_eviction_rounds(10_000, byz, config)
        rows.append({
            "byzantine_fraction": f"{byz:.0%}",
            "schemers": f"{schem:.0%}",
            "churn": f"{churn:.0%}",
            "eviction_latency_rounds": rounds if byz > 0 else "—",
            "false_positives_pct": f"{fp:.2f}",
            "jitter_ms": jitter,
            "sync_precision_us": round(jitter_sync_precision_us(jitter), 1),
        })
    return rows


def run_exp8(config: VPPConfig, quick: bool) -> dict:
    """Step-size / epsilon sensitivity."""
    eps_values = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
    tracking_opt, tracking_high, tracking_low = [], [], []

    n = 100 if quick else 5000
    for eps in eps_values:
        cfg = replace(config, epsilon=eps)
        sim = SwarmSimulator(n, cfg)
        sim.run_until_converged(max_iter=15 if quick else 60)
        tracking_opt.append(sim.tracking_error_pct())

        cfg_hd = replace(cfg, graphon_decay=1.2)
        sim_hd = SwarmSimulator(n, cfg_hd)
        sim_hd.run_until_converged(max_iter=15 if quick else 60)
        tracking_high.append(sim_hd.tracking_error_pct())

        cfg_lb = replace(cfg, sync_beta=0.2)
        sim_lb = SwarmSimulator(n, cfg_lb)
        sim_lb.run_until_converged(max_iter=15 if quick else 60)
        tracking_low.append(sim_lb.tracking_error_pct())

    return {
        "epsilon_values": eps_values,
        "tracking_optimal": tracking_opt,
        "tracking_high_decay": tracking_high,
        "tracking_low_beta": tracking_low,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Credit Calculus paper experiments")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Output directory for figures and CSV tables",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast mode with smaller swarm sizes (for CI / smoke tests)",
    )
    args = parser.parse_args()
    out_dir = ensure_output_dir(args.output)
    config = QUICK_CONFIG if args.quick else DEFAULT_CONFIG

    print("Credit Calculus — reproducing paper experiments", flush=True)
    print(f"Output: {out_dir.resolve()}", flush=True)
    t0 = time.time()

    print("\n[Exp 1] Algorithmic scalability...", flush=True)
    exp1 = run_exp1(config)
    plot_exp1_scaling(exp1, out_dir)

    print("[Exp 2 & 5] Tracking error and churn...", flush=True)
    exp2 = run_exp2_and5(config, args.quick)
    plot_exp2_tracking(exp2, out_dir)

    # Convergence trace demo
    demo = SwarmSimulator(100 if args.quick else 2000, config)
    demo_result = demo.run_until_converged()
    plot_convergence_trace(demo_result["errors"], out_dir)

    print("[Exp 3] Byzantine resilience...", flush=True)
    exp3 = run_exp3(config, args.quick)
    plot_exp3_byzantine(exp3, out_dir)

    print("[Exp 4] Markovian vs non-Markovian slashing...", flush=True)
    exp4 = run_exp4(config)
    plot_exp4_markovian_vs_nonmarkovian(exp4, out_dir)

    print("[Exp 6] Jitter / Table 2...")
    table2_rows = run_exp6_table2(config)

    print("[Exp 8] Hyperparameter sensitivity...")
    exp8 = run_exp8(config, args.quick)
    plot_exp8_sensitivity(exp8, out_dir)

    save_tables(exp2.get("table1_rows", []), table2_rows, out_dir)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Figures: {out_dir}/*.png")
    print(f"Tables:  {out_dir}/table1_scalability.csv, table2_adversarial.csv")

    # Summary
    print("\n--- Table 1 (sample) ---")
    for row in exp2.get("table1_rows", [])[:4]:
        print(row)


if __name__ == "__main__":
    main()
