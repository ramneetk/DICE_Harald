# LLM Experiments — Qwen3.5-0.8B Agents

LLM-driven reproduction of the Credit Calculus VPP experiments. Each agent is a **Qwen** instance (via vLLM; default target Qwen3.5-0.8B) that publishes a 24-hour JSON calendar; optional **Algorithm 1 guardrails** project raw LLM output onto the cooperative manifold.

## Setup

```bash
cd /homes/ramneet/DICE_Harald
source .venv/bin/activate
pip install -r requirements.txt
pip install -r llm_exp/requirements.txt
# vLLM (server only, separate env recommended):
# pip install vllm
```

## Run vLLM server

```bash
chmod +x llm_exp/scripts/serve_qwen.sh
llm_exp/scripts/serve_qwen.sh
```

## Run experiments

```bash
# Mock LLM (no GPU, CI smoke test)
python llm_exp/run_experiments.py --mock --quick --output llm_exp/results

# Live Qwen (vLLM must be running; auto-detects model from /v1/models)
python llm_exp/run_experiments.py --quick --n 10 25 --rounds 3 --output llm_exp/results

# Full scalability sweep
python llm_exp/run_experiments.py --quick --n 10 25 50 100 --rounds 8 --output llm_exp/results
```

## Three conditions

| Condition | Description |
|-----------|-------------|
| `baseline` | Numeric fair-share PGA ([`credit_calculus/swarm.py`](../credit_calculus/swarm.py)) |
| `llm_raw` | Qwen proposes calendar; no guardrail |
| `llm_guarded` | Qwen → Algorithm 1 guardrail ([`guardrail_apply.py`](../credit_calculus/guardrail_apply.py)) |

## Results summary

Verified **2025-06-25** with:

```bash
python llm_exp/run_experiments.py --quick --n 10 25 50 100 --rounds 8 --output llm_exp/results
```

**Run settings:** quick mode (δ=60 min → E=24), 8 planning rounds per condition, all agents are LLMs for `llm_raw` / `llm_guarded`. vLLM auto-detected model **`qwen3.5-9b-vllm`** (9B served locally; not Qwen3.5-0.8B). Total wall time ≈ **13.4 hours**.

### Tracking error (final L₂ %, lower is better)

| n | baseline (PGA) | llm_raw | llm_guarded |
|--:|---------------:|--------:|------------:|
| 10 | **3.19** | 100.0 | 100.0 |
| 25 | **3.19** | 100.0 | 100.0 |
| 50 | **3.19** | 87.45 | 93.14 |
| 100 | **3.19** | 88.22 | 91.67 |

The numeric **baseline** converges to ~3.2% tracking at every swarm size (same cooperative PGA as [`credit_calculus/`](../credit_calculus/)). **LLM conditions** stay near 87–100% error: agents mostly fail to produce valid JSON and fall back to idle calendars, so the swarm never coordinates on the grid attractor.

### JSON parse success & guardrail

| n | parse success (llm_raw) | parse success (llm_guarded) | guardrail modified |
|--:|------------------------:|----------------------------:|-------------------:|
| 10 | 0% | 0% | 0% |
| 25 | 0% | 0% | 0% |
| 50 | 4% | 1.3% | 1% |
| 100 | 5.1% | 4.0% | 4% |

Parse failures dominate: at n=10–25 the model returned **no** valid event-list JSON in 8×n calls. At n=50–100 a small fraction parsed (~1–5%), but schedules were still far from cooperative. Guardrails rarely fired because there was little valid LLM output to project.

### Latency & scalability

| n | mean LLM latency (ms/call) | wall time per n block (s) |
|--:|---------------------------:|--------------------------:|
| 10 | 8146 | 2608 (~44 min) |
| 25 | 8143 | 6516 (~109 min) |
| 50 | 8141 | 13037 (~217 min) |
| 100 | 8134 | 26070 (~434 min) |

Per-agent inference is **~8.1 s** (flat in n). Wall time scales **~linearly with n** because agents are queried sequentially (8 rounds × n agents × 2 LLM conditions per n block). Guardrail-only latency remains ~5 ms/agent (see numeric Exp 1 in root README).

### Takeaways

1. **Cooperative guardrails work in principle** (numeric baseline + Algorithm 1 in `credit_calculus/`) but **cannot rescue** swarms when LLMs do not emit parseable calendars.
2. **Prompt / parser / model size** need improvement before LLM agents match PGA (~3% vs ~90%+ tracking).
3. **Throughput:** parallel vLLM batching or a smaller/finetuned model would be needed for large-n sweeps at practical wall times.

### Root cause (log analysis, 2025-06-26)

Analysis of 3,029 JSONL records (`python llm_exp/scripts/analyze_logs.py`):

| Failure mode | Share of failures |
|--------------|------------------:|
| `no JSON object found` | ~83% |
| Truncated / malformed JSON | ~17% |

Live probes showed **Qwen3.5-9B** replies with a long `Thinking Process:` preamble instead of JSON when `response_format` is off. With JSON mode enabled, the model often **echoed the full input payload** in JSON, causing truncation at 256 tokens.

**Fixes applied** (see `parser.py`, `prompts.py`, `client.py`):

1. **`response_format={"type": "json_object"}`** on vLLM calls (config: `use_json_mode=True`)
2. **Compact user prompt** (~500 chars) asking for `{"events": [...]}` only
3. **Parser hardening** — strip thinking blocks, salvage truncated `events`, action aliases
4. **`temperature=0.0`**, **`max_tokens=384`**, **`raw_preview` in JSONL logs**

Validation after fixes: **10/10** parse success on live probes; quick rerun `--n 10 --rounds 2` → see `llm_exp/results_v2/`.

Full numbers: [`results/table_llm_summary.csv`](results/table_llm_summary.csv). Figures: `results/fig_llm_*.png`.

## Outputs

| File | Description |
|------|-------------|
| `fig_llm_tracking_vs_baseline.png` | Final tracking error by condition |
| `fig_llm_guardrail_effect.png` | Guardrail improvement over raw LLM |
| `fig_llm_parse_failures.png` | JSON parse failure rate |
| `fig_llm_latency.png` | Per-agent LLM + guardrail latency |
| `fig_llm_convergence.png` | Tracking error over planning rounds |
| `table_llm_summary.csv` | All runs summary |
| `logs/*.jsonl` | Per-agent audit trail (gitignored) |

## Tests

```bash
python -m pytest llm_exp/tests/ -q
```

## Swarm sizes

All agents are LLMs: **n ∈ {10, 25, 50, 100}** (default). Use `--n 10 25` to override.

LLM output uses compact **event-list JSON** (not full E×4 matrices):

```json
{"events": [{"start_hour": 17.25, "action": "DischargeV2G"}, ...]}
```

## Package layout

```
llm_exp/
  config.py       LLMConfig, PlanningCondition
  client.py       vLLM + MockLLMClient
  prompts.py      System/user prompts
  parser.py       JSON → Calendar
  agent.py        LLMAgent.plan()
  swarm.py        LLMSwarmSimulator
  plots.py        Figures
  run_experiments.py
```

Imports all VPP physics from [`credit_calculus/`](../credit_calculus/) — that package is not modified.
