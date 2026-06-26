# Credit Calculus — Experiment Reproduction

Python simulation of the experiments in **"The Credit Calculus: Cooperative Guardrails and Emergent Liveness in Agentic Swarms"** (Ruess & Shankar, SRI International, June 2026).

## Setup

```bash
cd /homes/ramneet/DICE_Harald
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run all experiments

Full reproduction (E=96, n up to 10⁵; may take 30+ minutes):

```bash
python run_experiments.py --output results_full
```

Quick validation run (E=24, n ≤ 500; ~15 seconds):

```bash
python run_experiments.py --quick --output results
```

## Reproducibility results

Verified on **2025-06-25** with:

```bash
python run_experiments.py --quick --output results
python -m pytest tests/ -q
```

**Runtime:** 14.5 s for all 8 experiments · **Unit tests:** 5/5 passed

### Table 1 — Scalability & tracking (`results/table1_scalability.csv`)

Quick-mode run (δ=60 min → E=24 intervals; flat VPP graphon):

| Swarm size n | E | Exec time (ms) | Memory (KB) | L₂ tracking error | Convergence steps |
|-------------:|--:|---------------:|------------:|------------------:|------------------:|
| 100 | 24 | 4.655 | 3.0 | **0.16%** | 15 |
| 200 | 24 | 4.671 | 3.0 | **0.16%** | 15 |
| 500 | 24 | 4.656 | 3.0 | **0.16%** | 15 |

Guardrail-only benchmark (Algorithm 1, independent of n):

| Calendar length E | Per-agent latency (ms) |
|------------------:|-----------------------:|
| 24 (quick mode) | 4.64 |
| 96 (paper setting) | 13.31 |

**Paper (Table 1, n=10²–10⁵, E=96):** exec ≈ 0.18 ms, memory ≈ 145 KB, tracking 0.31–0.94%, convergence ≈ 38–42 steps.

### Table 2 — Adversarial resilience (`results/table2_adversarial.csv`)

| Byzantine | Schemers | Churn | Eviction latency | False positives | Jitter | Sync offset (μs) |
|----------:|---------:|------:|-----------------:|----------------:|-------:|-----------------:|
| 0% | 0% | 0% | — | **0.00%** | 50 ms | 328.2 |
| 10% | 5% | 5% | 5 rounds | **0.00%** | 50 ms | 328.2 |
| 20% | 10% | 10% | 7 rounds | **0.00%** | 100 ms | 910.0 |
| 30% | 15% | 15% | 11 rounds | **0.00%** | 150 ms | 1663.4 |
| 40% | 20% | 20% | 16 rounds | **0.00%** | 200 ms | 2555.6 |
| 45% | 30% | 30% | 19 rounds | **0.00%** | 250 ms | 3567.6 |

**Paper (Table 2):** false-positive rate 0.00% at all loads; eviction latency 4–74 rounds at 10–45% Byzantine; sync precision 14–242 μs at 50–250 ms jitter.

### Qualitative findings (Figures 2–6)

| Experiment | Paper claim | Reproduced? |
|------------|-------------|:-----------:|
| **Exp 1** — Guardrail scaling | Flat O(1) per-agent latency vs n; O(n²) naive baseline grows | Yes — see `fig2_execution_scaling.png` |
| **Exp 2** — Renewable attractor | L₂ tracking error < 1% after ~40 iterations | Yes — 0.16% in quick run; see `fig3_tracking_error.png` |
| **Exp 3** — Byzantine nodes | 0% false-positive evictions; latency grows with Byzantine fraction | Yes — see `fig4_byzantine_eviction.png` |
| **Exp 4** — Free-rider schemers | Non-Markovian slashing drives ξᵢ → 0; Markovian regime pays out | Yes — deviation penalty 3600, payout ≡ 0; see `fig5_markovian_vs_nonmarkovian.png` |
| **Exp 5** — Network churn | Tracking degrades gracefully under 10–30% hourly churn | Yes — see `fig3_tracking_error.png` |
| **Exp 6** — Jitter tolerance | Sync offset grows sub-linearly with delay | Partial — trend matches; absolute μs values differ (see Notes) |
| **Exp 7** — Multi-goal guardrail | Bisection converges in 8–14 iterations | Implemented in Algorithm 1 bisection loop |
| **Exp 8** — Step-size stability | Divergence when η > c₀/Lt or ε → 0 | Yes — see `fig6_hyperparameter_sensitivity.png` |

Convergence trace for a 100-agent swarm: `results/convergence_trace.png`.

### Paper vs. this repository

| Metric | Paper | This repo | Match |
|--------|------:|----------:|:-----:|
| Per-agent guardrail scaling with n | Flat (~0.18 ms) | Flat (~4.7 ms) | Trend ✓, absolute ✗ |
| L₂ tracking error | 0.31–0.94% | 0.16% | ✓ |
| False-positive Byzantine evictions | 0.00% | 0.00% | ✓ |
| Schemer payout under non-Markovian slashing | ξᵢ ≡ 0 | ξᵢ = 0 | ✓ |
| Convergence iterations | ~38–42 | 15 (E=24) / ~40 expected (E=96) | Partial |
| Memory per node | ~145 KB | ~3 KB (quick) / ~12 KB (E=96) | Underestimate* |

\*Memory counts only calendar + gradient buffers, not full graphon neighborhood state.

## Outputs

| File | Paper reference |
|------|-----------------|
| `fig2_execution_scaling.png` | Figure 2 — guardrail latency vs *n* |
| `fig3_tracking_error.png` | Figure 3 — L₂ tracking error vs *n* and churn |
| `fig4_byzantine_eviction.png` | Figure 4 — Byzantine eviction latency |
| `fig5_markovian_vs_nonmarkovian.png` | Figure 5 — free-rider payout slashing |
| `fig6_hyperparameter_sensitivity.png` | Figure 6 — ε and graphon sensitivity |
| `convergence_trace.png` | PGA convergence trace |
| `table1_scalability.csv` | Table 1 |
| `table2_adversarial.csv` | Table 2 |

## Code layout

```
credit_calculus/
  config.py       # Paper constants (δ=15min, actions, ε, etc.)
  calendar.py     # Piecewise-constant async calendars
  graphon.py      # Watts-Strogatz + graphon topologies
  reward.py       # Sync probability, log-odds, Fréchet gradients
  credit.py       # Counterfactual / Shapley planning credit
  guardrail.py    # Algorithm 1 — ascent projection guardrail
  guardrail_apply.py  # Apply guardrail to agent calendars (Sections 8–9)
  swarm.py        # Swarm simulator + functional PGA
  adversarial.py  # Byzantine + non-Markovian slashing
  plots.py        # Figure generation
run_experiments.py
```

## What is implemented

- **Section 9 / Algorithm 1**: Simplex projection + bisection dual search guardrail
- **Section 4**: Log-odds reward, sync probability, counterfactual credit
- **Section 6**: Decentralized projected gradient ascent coordination
- **Section 7**: L₂ deviation penalty and Markovian vs non-Markovian payout
- **Section 10**: All eight experiment scenarios (scalability, tracking, Byzantine, schemers, churn, jitter, sensitivity)

## LLM agent experiments

Qwen agents via vLLM (baseline vs raw LLM vs guardrailed LLM) live in [`llm_exp/`](llm_exp/README.md).

```bash
pip install -r llm_exp/requirements.txt
python llm_exp/run_experiments.py --mock --quick --output llm_exp/results
```

Scalability results (n=10–100, qwen3.5-9b-vllm): see `llm_exp/results/table_llm_summary.csv`.

## Run tests

```bash
python -m pytest tests/ llm_exp/tests/ -q
```

## Notes

- **Guardrail latency:** Python implementation yields ~4.7 ms (E=24) / ~13 ms (E=96) per agent vs. ~0.18 ms reported in the paper (likely native edge deployment). The **scale-invariance with n** is reproduced.
- **Jitter sync precision:** Absolute microsecond offsets in Table 2 use a phenomenological model; trends match the paper but values are not calibrated to their hardware.
- **Swarm coordination** uses mean-field fair-share functional PGA (Section 6); guardrails for LLM schedules are in `guardrail_apply.py`.
- **Shapley credit** uses counterfactual credit + sampled coalitions rather than full functional PCE.
- **`--quick`** sets δ=60 min (E=24) for fast validation; omit it for the paper's δ=15 min (E=96) setting.
