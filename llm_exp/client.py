"""vLLM OpenAI-compatible client with mock fallback."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from llm_exp.config import LLMConfig, DEFAULT_LLM_CONFIG
from llm_exp.prompts import FEW_SHOT_EXAMPLE


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class MockLLMClient:
    """Deterministic cooperative schedule for tests and --mock runs."""

    def complete(self, system: str, user: str, seed: int = 0) -> LLMResponse:
        t0 = time.perf_counter()
        text = json.dumps(FEW_SHOT_EXAMPLE)
        elapsed = (time.perf_counter() - t0) * 1000
        return LLMResponse(
            text=text,
            prompt_tokens=len(user.split()),
            completion_tokens=len(text.split()),
            latency_ms=elapsed,
        )

    def complete_batch(
        self,
        messages: list[tuple[str, str]],
        seed: int = 0,
    ) -> list[LLMResponse]:
        return [self.complete(s, u, seed=seed + i) for i, (s, u) in enumerate(messages)]


class VLLMClient:
    def __init__(self, config: LLMConfig = DEFAULT_LLM_CONFIG):
        self.config = config
        from openai import OpenAI

        self._client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout_s,
        )

    def complete(self, system: str, user: str, seed: int | None = None) -> LLMResponse:
        seed = seed if seed is not None else self.config.seed
        t0 = time.perf_counter()
        kwargs: dict = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "seed": seed,
        }
        if self.config.use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        elapsed = (time.perf_counter() - t0) * 1000
        choice = resp.choices[0].message.content or ""
        usage = resp.usage
        return LLMResponse(
            text=choice,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_ms=elapsed,
        )

    def complete_batch(
        self,
        messages: list[tuple[str, str]],
        seed: int | None = None,
    ) -> list[LLMResponse]:
        return [
            self.complete(system, user, seed=(seed or self.config.seed) + i)
            for i, (system, user) in enumerate(messages)
        ]


def make_client(config: LLMConfig = DEFAULT_LLM_CONFIG):
    if config.use_mock:
        return MockLLMClient()
    return VLLMClient(config)


def server_reachable(config: LLMConfig = DEFAULT_LLM_CONFIG) -> bool:
    if config.use_mock:
        return True
    try:
        import urllib.request

        url = config.base_url.rstrip("/") + "/models"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False
