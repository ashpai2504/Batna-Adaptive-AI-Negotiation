"""OpenAI-compatible chat generators with a disk cache (Azure AI Foundry v1 endpoint, local servers)."""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time

from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, InternalServerError

from .config import ROOT

_GPT5_FAMILY = ("gpt-5", "o1", "o3", "o4")


def load_env(path=os.path.join(ROOT, ".env")):
    env = {}
    if os.path.exists(path):
        for line in open(path):
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.strip().split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    env.update({k: v for k, v in os.environ.items() if k.startswith("AZURE_AI_")})
    return env


class ChatGenerator:
    """Callable (messages, temperature, top_p) -> text, matching rlvr-batna's generator interface.

    gpt-5 family models on Azure reject non-default temperature/top_p; we send reasoning_effort instead
    and record that deviation from rlvr-batna's T=0.7 protocol in the run metadata.
    """

    def __init__(self, model: str, base_url: str, api_key: str, label: str | None = None, max_tokens: int = 512,
                 reasoning_effort: str | None = "minimal", cache_dir: str | None = None, timeout: float = 120):
        self.model, self.label = model, label or model
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.is_reasoning = any(model.startswith(p) for p in _GPT5_FAMILY)
        self.cache_dir = cache_dir or os.path.join(ROOT, "results", "dynbatna", "cache", _slug(label or model))
        os.makedirs(self.cache_dir, exist_ok=True)
        self.usage = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._lock = threading.Lock()

    def settings(self) -> dict:
        return {"model": self.model, "max_tokens": self.max_tokens,
                "reasoning_effort": self.reasoning_effort if self.is_reasoning else None,
                "sampling": "provider default (temperature=1) for reasoning models" if self.is_reasoning else "T/top_p as passed"}

    def __call__(self, messages, temperature=0.7, top_p=1.0, seed_key="") -> str:
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        kwargs = {"model": self.model, "messages": msgs}
        if self.is_reasoning:
            kwargs["max_completion_tokens"] = self.max_tokens
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs.update(max_tokens=self.max_tokens, temperature=temperature, top_p=top_p)
        key = hashlib.sha256(json.dumps([kwargs, seed_key], sort_keys=True).encode()).hexdigest()
        path = os.path.join(self.cache_dir, key[:2], key + ".json")
        if os.path.exists(path):
            with self._lock:
                self.usage["cached"] += 1
            return json.load(open(path))["content"]
        for attempt in range(6):
            try:
                r = self.client.chat.completions.create(**kwargs)
                break
            except (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError) as e:
                if attempt == 5:
                    raise
                time.sleep(min(60, 2 * 2 ** attempt))
        content = (r.choices[0].message.content or "").strip()
        with self._lock:
            self.usage["calls"] += 1
            if r.usage:
                self.usage["prompt_tokens"] += r.usage.prompt_tokens or 0
                self.usage["completion_tokens"] += r.usage.completion_tokens or 0
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump({"content": content}, open(path, "w"))
        return content


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def azure_generator(label="gpt-5-mini", model=None, **kw) -> ChatGenerator:
    env = load_env()
    return ChatGenerator(model or env.get("AZURE_AI_DEPLOYMENT", "gpt-5-mini"), env["AZURE_AI_ENDPOINT"],
                         env["AZURE_AI_API_KEY"], label=label, **kw)


def local_generator(model: str, label: str | None = None, base_url="http://localhost:8080/v1", **kw) -> ChatGenerator:
    """mlx_lm.server / Ollama / vLLM OpenAI-compatible endpoints."""
    return ChatGenerator(model, base_url, "local", label=label or model, reasoning_effort=None, **kw)
