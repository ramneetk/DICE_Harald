# LLM Experiments — Qwen3.5-0.8B Agents

LLM-driven reproduction of the Credit Calculus VPP experiments. Each agent is a **Qwen3.5-0.8B** instance (via vLLM) that publishes a 24-hour JSON calendar; optional **Algorithm 1 guardrails** project raw LLM output onto the cooperative manifold.

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
