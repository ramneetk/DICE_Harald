"""Generate paper figures (Figures 2–6, Tables 1–2)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_exp1_scaling(results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ns = results["n_values"]
    flat = results["flat_ms"]
    nested = results["nested_ms"]
    hier = results["hierarchical_ms"]
    naive = [0.01 * n ** 2 / 1000 for n in ns]  # illustrative O(n^2) baseline

    ax.loglog(ns, flat, "o-", label="Flat Graphon VPP")
    ax.loglog(ns, nested, "s-", label="Nested Grid Graphon")
    ax.loglog(ns, hier, "^-", label="Hierarchical Graphon")
    ax.loglog(ns, naive, "k--", alpha=0.5, label="Naive Global O(n²)")
    ax.set_xlabel("Network Size n")
    ax.set_ylabel("Per-Agent Execution Time (ms)")
    ax.set_title("Execution Scaling per Graphon (Exp. 1)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_execution_scaling.png", dpi=150)
    plt.close(fig)


def plot_exp2_tracking(results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ns = results["n_values"]
    ax.semilogx(ns, results["tracking_0_churn"], "o-", label="VPP Flat (0% Churn)")
    ax.semilogx(ns, results["tracking_10_churn"], "s-", label="VPP with 10% Churn")
    ax.semilogx(ns, results["tracking_30_churn"], "^-", label="VPP with 30% Churn")
    ax.set_xlabel("Swarm Size n")
    ax.set_ylabel("Tracking Error L₂ (% Offset)")
    ax.set_title("Tracking Error and Churn (Exp. 2 & 5)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_tracking_error.png", dpi=150)
    plt.close(fig)


def plot_exp3_byzantine(results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    fracs = results["byzantine_pct"]
    ax.plot(fracs, results["eviction_rounds"], "o-", label="Eviction rounds (θ_evict)")
    ax2 = ax.twinx()
    ax2.plot(fracs, results["false_positives"], "s--", color="orange", label="False Positives")
    ax.set_xlabel("Byzantine Fraction (%)")
    ax.set_ylabel("Eviction Latency (rounds)")
    ax2.set_ylabel("False Positive Rate (%)")
    ax.set_title("Byzantine Eviction Metrics (Exp. 3)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_byzantine_eviction.png", dpi=150)
    plt.close(fig)


def plot_exp4_markovian_vs_nonmarkovian(results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rates = results["deception_rates"]
    ax.plot(rates, results["markovian_payout"], "o-", label="Markovian Payout (No Defense)")
    ax.plot(rates, results["nonmarkovian_payout"], "s-", label="Non-Markovian (Deviation Penalty)")
    ax.set_xlabel("Deception Rate (%)")
    ax.set_ylabel("Slashed Net Payout ξᵢ (norm.)")
    ax.set_title("Markovian vs. Non-Markovian Regimes (Exp. 4)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_markovian_vs_nonmarkovian.png", dpi=150)
    plt.close(fig)


def plot_exp8_sensitivity(results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    eps = results["epsilon_values"]
    ax.plot(eps, results["tracking_optimal"], "o-", label="RBF Decay γ = 0.5 (Optimal)")
    ax.plot(eps, results["tracking_high_decay"], "s-", label="High Decay γ = 1.2")
    ax.plot(eps, results["tracking_low_beta"], "^-", label="Low Temperature β = 0.2")
    ax.set_xlabel("Log-Odds Smoothing Parameter ε")
    ax.set_ylabel("Tracking Error (L₂ % Offset)")
    ax.set_title("Hyperparameter Sensitivity (Exp. 6 & 8)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig6_hyperparameter_sensitivity.png", dpi=150)
    plt.close(fig)


def plot_convergence_trace(errors: list[float], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(errors) + 1), errors, "-")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Tracking Error L₂ (% Offset)")
    ax.set_title("Swarm Convergence Trace")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_trace.png", dpi=150)
    plt.close(fig)


def save_tables(table1_rows: list[dict], table2_rows: list[dict], out_dir: Path) -> None:
    import csv

    with open(out_dir / "table1_scalability.csv", "w", newline="") as f:
        if table1_rows:
            writer = csv.DictWriter(f, fieldnames=table1_rows[0].keys())
            writer.writeheader()
            writer.writerows(table1_rows)

    with open(out_dir / "table2_adversarial.csv", "w", newline="") as f:
        if table2_rows:
            writer = csv.DictWriter(f, fieldnames=table2_rows[0].keys())
            writer.writeheader()
            writer.writerows(table2_rows)
