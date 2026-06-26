"""Plots for LLM experiment results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_tracking_comparison(results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ns = results["n_values"]
    for key, label in [
        ("baseline", "Baseline (PGA)"),
        ("llm_raw", "LLM raw"),
        ("llm_guarded", "LLM + guardrail"),
    ]:
        if key in results:
            ax.plot(ns, results[key], "o-", label=label)
    ax.set_xlabel("Swarm size n")
    ax.set_ylabel("Final L₂ tracking error (%)")
    ax.set_title("LLM vs Baseline Tracking (Exp 2)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_llm_tracking_vs_baseline.png", dpi=150)
    plt.close(fig)


def plot_convergence(results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label in [
        ("baseline_errors", "Baseline"),
        ("llm_raw_errors", "LLM raw"),
        ("llm_guarded_errors", "LLM + guardrail"),
    ]:
        if key in results and results[key]:
            ax.plot(range(1, len(results[key]) + 1), results[key], label=label)
    ax.set_xlabel("Planning iteration")
    ax.set_ylabel("L₂ tracking error (%)")
    ax.set_title("Convergence trace")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_llm_convergence.png", dpi=150)
    plt.close(fig)


def plot_guardrail_effect(results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ns = results.get("n_values", [])
    raw = results.get("llm_raw", [])
    guarded = results.get("llm_guarded", [])
    if raw and guarded:
        improvement = [r - g for r, g in zip(raw, guarded)]
        ax.bar([str(n) for n in ns], improvement, color="steelblue")
    ax.set_xlabel("Swarm size n")
    ax.set_ylabel("Tracking error reduction (raw − guarded) %")
    ax.set_title("Guardrail effect on LLM schedules")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_llm_guardrail_effect.png", dpi=150)
    plt.close(fig)


def plot_parse_failures(results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ns = results.get("n_values", [])
    rates = results.get("parse_success_rate", [])
    if ns and rates:
        fail = [100 * (1 - r) for r in rates]
        ax.plot(ns, fail, "s-", color="coral")
    ax.set_xlabel("Swarm size n")
    ax.set_ylabel("Parse failure rate (%)")
    ax.set_title("LLM JSON parse failures")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_llm_parse_failures.png", dpi=150)
    plt.close(fig)


def plot_latency(results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ns = results.get("n_values", [])
    llm_ms = results.get("llm_latency_ms", [])
    guard_ms = results.get("guardrail_ms", [])
    if ns and llm_ms:
        ax.plot(ns, llm_ms, "o-", label="LLM inference")
    if ns and guard_ms:
        ax.plot(ns, guard_ms, "s-", label="Guardrail only")
    ax.set_xlabel("Swarm size n")
    ax.set_ylabel("Per-agent latency (ms)")
    ax.set_title("Exp 1: LLM + guardrail latency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_llm_latency.png", dpi=150)
    plt.close(fig)


def save_summary_csv(rows: list[dict], out_dir: Path, filename: str = "table_llm_summary.csv") -> None:
    import csv

    if not rows:
        return
    path = out_dir / filename
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_steering_summary(results: list[dict], out_dir: Path) -> None:
    """Bar chart of tracking error by run_id."""
    if not results:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    run_ids = [r["run_id"] for r in results]
    tracking = [r["tracking_pct"] for r in results]
    colors = ["forestgreen" if r.get("use_baseline") else "steelblue" for r in results]
    ax.bar(run_ids, tracking, color=colors)
    ax.set_xlabel("Run ID")
    ax.set_ylabel("Final L₂ tracking error (%)")
    ax.set_title("Steering experiment matrix (R0–R6)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_steering_summary.png", dpi=150)
    plt.close(fig)


def plot_persona_sweep(results: list[dict], out_dir: Path) -> None:
    """Tracking vs persona for LLM runs."""
    llm_runs = [r for r in results if not r.get("use_baseline")]
    if not llm_runs:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    personas = [r["persona"] for r in llm_runs]
    tracking = [r["tracking_pct"] for r in llm_runs]
    ax.bar(personas, tracking, color="coral")
    ax.set_xlabel("Persona")
    ax.set_ylabel("Final tracking error (%)")
    ax.set_title("Persona sweep — tracking error")
    plt.xticks(rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_persona_sweep.png", dpi=150)
    plt.close(fig)


def plot_guardrail_pareto(results: list[dict], out_dir: Path) -> None:
    """Guardrail mod rate vs tracking."""
    llm_runs = [r for r in results if not r.get("use_baseline")]
    if not llm_runs:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    mod_rates = [r.get("guardrail_mod_rate", 0) for r in llm_runs]
    tracking = [r["tracking_pct"] for r in llm_runs]
    labels = [r["run_id"] for r in llm_runs]
    ax.scatter(mod_rates, tracking, s=80, c="steelblue")
    for label, x, y in zip(labels, mod_rates, tracking):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Guardrail modification rate")
    ax.set_ylabel("Final tracking error (%)")
    ax.set_title("Guardrail Pareto: mod rate vs tracking")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_guardrail_pareto.png", dpi=150)
    plt.close(fig)


def plot_feedback_ablation(results: list[dict], out_dir: Path) -> None:
    """Tracking by feedback mode."""
    llm_runs = [r for r in results if not r.get("use_baseline")]
    if not llm_runs:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    modes = [r["feedback_mode"] for r in llm_runs]
    tracking = [r["tracking_pct"] for r in llm_runs]
    ax.bar(modes, tracking, color="mediumpurple")
    ax.set_xlabel("Feedback mode")
    ax.set_ylabel("Final tracking error (%)")
    ax.set_title("Feedback ablation")
    plt.xticks(rotation=25, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_feedback_ablation.png", dpi=150)
    plt.close(fig)
