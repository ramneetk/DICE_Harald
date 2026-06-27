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
  guardrail_extended.py  # attractor_seek, hybrid_band (steering experiments)
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

There are two LLM experiment tracks in [`llm_exp/`](llm_exp/README.md):

| Track | Script | What it tests |
|-------|--------|---------------|
| **Original scalability** | `llm_exp/run_experiments.py` | Baseline PGA vs raw LLM vs guardrailed LLM at n=10–100 |
| **Steering (safety bubble)** | `llm_exp/run_steering_experiments.py` | Persona, guardrail mode, and credit feedback to push agents toward mission success |

Install LLM extras once:

```bash
pip install -r llm_exp/requirements.txt
```

### Original LLM conditions (quick start)

```bash
# Offline — no GPU, returns a fixed cooperative schedule (~seconds)
python llm_exp/run_experiments.py --mock --quick --output llm_exp/results

# Live — requires vLLM (see below)
python llm_exp/run_experiments.py --quick --n 10 25 --rounds 8 --output llm_exp/results
```

Scalability results: `llm_exp/results/table_llm_summary.csv`. Details: [`llm_exp/README.md`](llm_exp/README.md).

---

### Steering experiments — pushing agents outside the “safety bubble”

**In plain terms:** Each simulated EV battery is an LLM that publishes a 24-hour charging/discharging schedule (JSON). Out of the box, LLMs tend to be **conservative** — they idle, parse poorly, or under-contribute — while the numeric simulator (PGA) tracks the grid target to ~**3%** error. The steering suite asks: *can we change the agent’s **persona**, add smarter **guardrails**, and show **credit feedback** from prior rounds so LLM swarms coordinate better?*

This is **task-level** risk appetite (grid tracking vs battery comfort) in a **simulated** VPP — not bypassing chat safety filters.

#### Three knobs you can tune

| Knob | Options | Effect |
|------|---------|--------|
| **Persona** | `conservative`, `cooperative`, `aggressive`, `mission_critical`, `byzantine` | System prompt that sets risk appetite (`llm_exp/personas.py`) |
| **Guardrail mode** | `none`, `floor_only`, `attractor_seek`, `hybrid_band` | How Algorithm 1 projects bad schedules (`credit_calculus/guardrail_extended.py`) |
| **Feedback mode** | `none`, `credit_only`, `gradient_only`, `credit_gradient`, `full` | What prior-round info goes back into the user prompt (`llm_exp/prompts.py`) |

- **`floor_only`** — current paper guardrail: enforce a cooperative *minimum* (pull under-contributors up).
- **`attractor_seek`** — also push toward the fair-share ideal, not just the floor.
- **`hybrid_band`** — floor *and* ceiling: bounded over-commitment during grid shocks.
- **Feedback** — Shapley credit, gradient hints (“use DischargeV2G at 17h”), and non-Markovian slashing warnings.

#### The R0–R6 experiment matrix

One command runs all seven preset configurations (8 planning rounds each):

| Run | Persona | Guardrail | Feedback | Swarm n | Role |
|-----|---------|-----------|----------|--------:|------|
| **R0** | cooperative | none | none | 10 | Numeric PGA baseline (no LLM) |
| **R1** | cooperative | floor_only | none | 10 | Original guarded LLM control |
| **R2** | aggressive | floor_only | none | 10 | Persona-only steering |
| **R3** | aggressive | attractor_seek | gradient | 10 | Persona + attractor guardrail + hints |
| **R4** | mission_critical | hybrid_band | credit+gradient | 10 | High-risk persona + band guardrail |
| **R5** | aggressive | attractor_seek | full | 10 | Full steering stack |
| **R6** | aggressive | attractor_seek | full | 25 | Scale-up of R5 |

**Success target (v1):** any LLM run with **< 50%** L₂ tracking at n=10 (vs ~227% in early guarded runs). Stretch goal: approach numeric **~3%**.

#### How to run (step by step)

**1. Offline smoke test** — validates the pipeline in a few seconds, no GPU:

```bash
python llm_exp/run_steering_experiments.py \
  --mock --quick --runs all \
  --output llm_exp/results_steering_smoke
```

Uses a mock LLM that always returns the same cooperative JSON. Good for CI and sanity checks; tracking numbers for LLM runs will look artificially high (~260%) because every agent emits the identical schedule.

**2. Live Qwen** — needs a vLLM server:

```bash
# Terminal 1 — start server (once)
chmod +x llm_exp/scripts/serve_qwen.sh
llm_exp/scripts/serve_qwen.sh

# Terminal 2 — full matrix (~1–2 hours at n=10–25, ~8 s per LLM call)
python llm_exp/run_steering_experiments.py \
  --quick --runs all \
  --output llm_exp/results_steering
```

The runner auto-detects the model from `http://127.0.0.1:8000/v1/models` (e.g. `qwen3.5-9b-vllm`).

**Run a subset** — comma-separated run IDs:

```bash
python llm_exp/run_steering_experiments.py --mock --quick --runs R0,R3,R5
```

**Long-running jobs in tmux** (survives disconnect):

```bash
tmux new-session -d -s steering_live \
  "cd $(pwd) && source .venv/bin/activate && \
   python llm_exp/run_steering_experiments.py --quick --runs all \
   --output llm_exp/results_steering 2>&1 | tee llm_exp/results_steering/run.log"

tmux attach -t steering_live    # watch progress
# detach: Ctrl-b then d
tail -f llm_exp/results_steering/run.log
```

#### Where to find results

| Output | Location |
|--------|----------|
| Summary table | `llm_exp/results_steering/table_steering_summary.csv` |
| Run log | `llm_exp/results_steering/run.log` |
| Per-agent audit trail | `llm_exp/results_steering/logs/R{n}_n{size}.jsonl` |
| Tracking by run | `fig_steering_summary.png` |
| Persona comparison | `fig_persona_sweep.png` |
| Guardrail mod rate vs error | `fig_guardrail_pareto.png` |
| Feedback ablation | `fig_feedback_ablation.png` |

**Key columns in the CSV:** `tracking_pct` (lower is better), `parse_success_rate`, `guardrail_mod_rate`, `converged`.

#### Mock smoke results (reference)

Offline run (`results_steering_smoke/`, mock LLM, quick mode):

| Run | tracking_pct | guardrail_mod_rate |
|-----|-------------:|-------------------:|
| R0 (PGA baseline) | **3.19** | 0.0 |
| R1–R5 (LLM) | ~267 | 0.54–1.0 |
| R6 (n=25) | 257.76 | 0.69 |

R0 confirms the numeric coordinator works; LLM rows reflect the mock’s identical schedule, not real Qwen behavior.

#### Live steering results (qwen-analysis)

Verified **2026-06-27** with:

```bash
python llm_exp/run_steering_experiments.py --quick --runs all --output llm_exp/results_steering
```

**Run settings:** quick mode (δ=60 min → E=24), 8 planning rounds per run, vLLM model **`qwen-analysis`**, total wall time ≈ **19 min**. JSON parse success **100%** on all LLM runs.

##### Tracking error (final L₂ %, lower is better)

| Run | Persona | Guardrail | Feedback | n | tracking_pct | guardrail_mod_rate |
|-----|---------|-----------|----------|--:|-------------:|-------------------:|
| **R0** | cooperative (PGA) | none | none | 10 | **3.19** | 0.0 |
| R1 | cooperative | floor_only | none | 10 | 290.83 | 0.84 |
| R2 | aggressive | floor_only | none | 10 | 290.71 | 0.88 |
| R3 | aggressive | attractor_seek | gradient | 10 | 264.44 | 0.55 |
| **R4** | mission_critical | hybrid_band | credit+gradient | 10 | **195.31** | 0.99 |
| R5 | aggressive | attractor_seek | full | 10 | 458.78 | 0.58 |
| R6 | aggressive | attractor_seek | full | 25 | 294.15 | 0.60 |

Full table and figures: [`llm_exp/results_steering/`](llm_exp/results_steering/).

##### Takeaways

- **Numeric PGA (R0)** still wins decisively at **3.19%** tracking — the LLM gap remains large.
- **Best LLM run: R4** (mission-critical persona + hybrid band + credit/gradient feedback) at **195%** — ~33% better than R1/R2 (~291%), but far from the v1 target of **< 50%**.
- **Persona alone (R2 vs R1)** did not help; aggressive and cooperative prompts performed similarly.
- **Attractor seek + gradient hints (R3)** improved modestly over floor-only (264% vs 291%).
- **Full feedback stack (R5)** hurt badly (**459%**) — slashing + attractor seek can destabilize coordination.
- **Scale-up (R6, n=25)** did not improve on R5; tracking stayed ~294%.

The v1 success criterion (any LLM run **< 50%** at n=10) was **not met**. R4 is the most promising configuration for follow-up sweeps (floor fraction, temperature, more rounds).

#### Steering code layout

```
llm_exp/
  personas.py                 # conservative → mission_critical prompts
  prompts.py                  # user prompts + FeedbackContext blocks
  agent.py                    # LLMAgent.plan() with persona + guardrail mode
  steering_swarm.py           # SteeringSwarmSimulator + R0–R6 matrix
  run_steering_experiments.py # matrix runner + plots
credit_calculus/
  guardrail_extended.py       # attractor_seek, hybrid_band modes
```

## Run tests

```bash
python -m pytest tests/ llm_exp/tests/ -q   # includes steering / persona / guardrail tests
```

## Notes

- **Guardrail latency:** Python implementation yields ~4.7 ms (E=24) / ~13 ms (E=96) per agent vs. ~0.18 ms reported in the paper (likely native edge deployment). The **scale-invariance with n** is reproduced.
- **Jitter sync precision:** Absolute microsecond offsets in Table 2 use a phenomenological model; trends match the paper but values are not calibrated to their hardware.
- **Swarm coordination** uses mean-field fair-share functional PGA (Section 6); guardrails for LLM schedules are in `guardrail_apply.py`.
- **Shapley credit** uses counterfactual credit + sampled coalitions rather than full functional PCE.
- **`--quick`** sets δ=60 min (E=24) for fast validation; omit it for the paper's δ=15 min (E=96) setting.
