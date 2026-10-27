"""Minimal Azure AI Foundry / Azure OpenAI chat-completions client with tool calling and a disk cache.

Reads .env (never committed):
  AZURE_AI_API_KEY      resource key
  AZURE_AI_ENDPOINT     https://<resource>.services.ai.azure.com/  (or .openai.azure.com)
  AZURE_AI_DEPLOYMENT   deployment name, e.g. gpt-5-mini
  AZURE_AI_API_VERSION  optional, default 2025-01-01-preview
  AZURE_AI_REASONING    optional reasoning_effort for reasoning models (minimal|low|medium|high)
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env(path=os.path.join(ROOT, ".env")):
    env = {}
    if os.path.exists(path):
        for line in open(path):
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.strip().split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return {**env, **{k: v for k, v in os.environ.items() if k.startswith("AZURE_AI_")}}


class AzureChat:
    def __init__(self, deployment=None, cache_dir=os.path.join(ROOT, "results", "llm", "cache"),
                 max_completion_tokens=2500, timeout=120):
        env = load_env()
        self.key = env["AZURE_AI_API_KEY"]
        self.endpoint = env["AZURE_AI_ENDPOINT"].rstrip("/")
        self.deployment = deployment or env.get("AZURE_AI_DEPLOYMENT", "gpt-5-mini")
        self.api_version = env.get("AZURE_AI_API_VERSION", "2025-01-01-preview")
        self.reasoning = env.get("AZURE_AI_REASONING", "low")
        self.max_tokens = max_completion_tokens
        self.timeout = timeout
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.usage = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._lock = threading.Lock()

    @property
    def url(self):
        return f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"

    def chat(self, messages, tools=None, seed_key=""):
        body = {"messages": messages, "max_completion_tokens": self.max_tokens}
        if self.reasoning:
            body["reasoning_effort"] = self.reasoning
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        key = hashlib.sha256(json.dumps([self.deployment, body, seed_key], sort_keys=True).encode()).hexdigest()
        path = os.path.join(self.cache_dir, key[:2], key + ".json")
        if os.path.exists(path):
            with self._lock:
                self.usage["cached"] += 1
            return json.load(open(path))
        data = self._post(body)
        msg = data["choices"][0]["message"]
        u = data.get("usage", {})
        with self._lock:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
            self.usage["completion_tokens"] += u.get("completion_tokens", 0)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(msg, open(path, "w"))
        return msg

    def _post(self, body, retries=6):
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"api-key": self.key, "Content-Type": "application/json"})
        for k in range(retries):
            try:
                return json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())
            except urllib.error.HTTPError as e:
                detail = e.read()[:400].decode(errors="ignore")
                if e.code in (429, 500, 502, 503, 504) and k < retries - 1:
                    time.sleep(min(60, 4 * 2 ** k))
                    continue
                raise RuntimeError(f"Azure HTTP {e.code}: {detail}") from None
            except (urllib.error.URLError, TimeoutError):
                if k == retries - 1:
                    raise
                time.sleep(4 * 2 ** k)


class AzureV1Chat:
    """Azure AI Foundry OpenAI v1 endpoint (<resource>/openai/v1/chat/completions); text-only .chat() like OllamaChat."""

    def __init__(self, cache_dir=os.path.join(ROOT, "results", "llm", "cache_azure_v1"), max_tokens=1200,
                 timeout=120):
        env = load_env()
        self.key = env["AZURE_AI_API_KEY"]
        self.url = env["AZURE_AI_ENDPOINT"].rstrip("/") + "/chat/completions"
        self.model = env.get("AZURE_AI_DEPLOYMENT", "gpt-5-mini")
        self.reasoning = env.get("AZURE_AI_REASONING", "minimal")
        self.max_tokens, self.timeout, self.cache_dir = max_tokens, timeout, cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.usage = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._lock = threading.Lock()

    def chat(self, messages, seed=0, seed_key=""):
        body = {"model": self.model, "messages": messages, "max_completion_tokens": self.max_tokens,
                "reasoning_effort": self.reasoning}
        key = hashlib.sha256(json.dumps([body, seed, seed_key], sort_keys=True).encode()).hexdigest()
        path = os.path.join(self.cache_dir, key[:2], key + ".json")
        if os.path.exists(path):
            with self._lock:
                self.usage["cached"] += 1
            return json.load(open(path))["content"]
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"api-key": self.key, "Content-Type": "application/json"})
        for k in range(6):
            try:
                data = json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and k < 5:
                    time.sleep(min(30, 2 * 2 ** k))
                    continue
                raise RuntimeError(f"Azure HTTP {e.code}: {e.read()[:300]}") from None
            except (urllib.error.URLError, TimeoutError):
                if k == 5:
                    raise
                time.sleep(2 * 2 ** k)
        content = data["choices"][0]["message"].get("content") or ""
        u = data.get("usage", {})
        with self._lock:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
            self.usage["completion_tokens"] += u.get("completion_tokens", 0)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump({"content": content}, open(path, "w"))
        return content


class OllamaChat:
    """Local open-weights model served by Ollama (http://localhost:11434). Same cached .chat() interface."""

    def __init__(self, model="qwen3:4b-instruct-2507-q4_K_M", host="http://localhost:11434",
                 cache_dir=os.path.join(ROOT, "results", "llm", "cache_ollama"),
                 num_predict=220, num_ctx=4096, temperature=0.7, timeout=300):
        self.model, self.host = model, host.rstrip("/")
        self.options = {"num_predict": num_predict, "num_ctx": num_ctx, "temperature": temperature}
        self.timeout = timeout
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.usage = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    def chat(self, messages, seed=0, seed_key=""):
        body = {"model": self.model, "messages": messages, "stream": False, "think": False,
                "options": {**self.options, "seed": int(seed)}}
        key = hashlib.sha256(json.dumps([body, seed_key], sort_keys=True).encode()).hexdigest()
        path = os.path.join(self.cache_dir, key[:2], key + ".json")
        if os.path.exists(path):
            self.usage["cached"] += 1
            return json.load(open(path))["content"]
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        for k in range(4):
            try:
                data = json.loads(urllib.request.urlopen(req, timeout=self.timeout).read())
                break
            except (urllib.error.URLError, TimeoutError):
                if k == 3:
                    raise
                time.sleep(5 * 2 ** k)
        content = data["message"]["content"]
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += data.get("prompt_eval_count", 0)
        self.usage["completion_tokens"] += data.get("eval_count", 0)
        self.usage["seconds"] += data.get("total_duration", 0) / 1e9
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump({"content": content}, open(path, "w"))
        return content
