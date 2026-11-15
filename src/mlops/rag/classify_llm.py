from __future__ import annotations

import os
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from openai import APIStatusError, OpenAI, RateLimitError

_LAST_CALL_TS = 0.0


@lru_cache(maxsize=None)
def _client(base_url: str, key_env: str) -> OpenAI:
    api_key = os.environ.get(key_env)
    if not api_key:
        raise RuntimeError(f"Environment variable {key_env} is not set")
    return OpenAI(base_url=base_url, api_key=api_key)


def render(prompt_path: str, **kw) -> str:
    text = Path(prompt_path).read_text(encoding="utf-8")
    for k, v in kw.items():
        text = text.replace("{{" + k + "}}", str(v))
    return text


def _retry_delay(err: Exception, attempt: int) -> float:
    m = re.search(r"retry in (\d+(?:\.\d+)?)s", str(err), re.I)
    if m:
        return float(m.group(1)) + 1.5
    return min(15.0 * (attempt + 1), 65.0)


def chat(section: dict, prompt: str, max_retries: int = 8) -> dict:
    """section = cfg['llm'] або cfg['judge']."""
    global _LAST_CALL_TS
    c = _client(section["base_url"], section["api_key_env"])

    kwargs = {
        "model": section["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": section["temperature"],
        "max_tokens": max(section["max_tokens"], 1024),
    }
    if "gemini" in section["model"].lower():
        kwargs["reasoning_effort"] = "low"

    # На безкоштовному тарифі (price == 0) тримаємо темп <= 15 RPM (~1 запит / 4.1 с)
    min_interval = 4.1 if section.get("price_per_1m_input_usd", 0) == 0 else 0.0

    for attempt in range(max_retries):
        if min_interval > 0:
            wait = min_interval - (time.perf_counter() - _LAST_CALL_TS)
            if wait > 0:
                time.sleep(wait)
        try:
            t0 = time.perf_counter()
            r = c.chat.completions.create(**kwargs)
            dt = time.perf_counter() - t0
            _LAST_CALL_TS = time.perf_counter()
            u = r.usage
            return {
                "text": r.choices[0].message.content or "",
                "latency": dt,
                "tin": u.prompt_tokens if u else 0,
                "tout": u.completion_tokens if u else 0,
            }
        except (RateLimitError, APIStatusError) as e:
            _LAST_CALL_TS = time.perf_counter()
            if isinstance(e, APIStatusError) and e.status_code not in (429, 500, 503):
                raise
            if attempt == max_retries - 1:
                raise
            delay = _retry_delay(e, attempt)
            print(f"[rate-limit] waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(delay)

    raise RuntimeError("Failed to get response from LLM API")


def parse_score(text: str, key: str) -> float | None:
    m = re.search(rf'"{key}"\s*:\s*([0-9]*\.?[0-9]+)', text)
    return float(m.group(1)) if m else None


class Meter:
    def __init__(self) -> None:
        self.lat: list[float] = []
        self.tin = 0
        self.tout = 0

    def add(self, r: dict) -> None:
        self.lat.append(r["latency"])
        self.tin += r["tin"]
        self.tout += r["tout"]

    def summary(self, section: dict, prefix: str) -> dict:
        p50, p95 = np.percentile(self.lat, [50, 95])
        cost = (self.tin / 1e6) * section["price_per_1m_input_usd"] + \
               (self.tout / 1e6) * section["price_per_1m_output_usd"]
        return {
            f"{prefix}_calls": len(self.lat),
            f"{prefix}_tokens_in": self.tin,
            f"{prefix}_tokens_out": self.tout,
            f"{prefix}_cost_usd": cost,
            f"{prefix}_latency_p50_s": float(p50),
            f"{prefix}_latency_p95_s": float(p95),
            f"{prefix}_latency_first_call_s": self.lat[0],
            f"{prefix}_compute_seconds": float(sum(self.lat)),
        }