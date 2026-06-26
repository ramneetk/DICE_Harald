#!/usr/bin/env bash
# Start vLLM with Qwen3.5-0.8B (adjust model path as needed).
set -euo pipefail
MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
PORT="${PORT:-8000}"
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --dtype auto
