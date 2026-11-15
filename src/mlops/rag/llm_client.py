from __future__ import annotations

import os
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from openai import OpenAI


@lru_cache(maxsize=None)
def _client(base_url: str, key_env: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=os.environ.get(key_env) or "EMPTY")


def render(prompt_path: str, **kw) -> str:
    text = Path(prompt_path).read_text()
    for k, v in kw.items():
        text = text.replace("{{" + k + "}}", str(v))
    return text


def chat(section: dict, prompt: str) -> dict:
    c = _client(section["base_url"], section["api_key_env"])
    t0 = time.perf_counter()
    r = c.chat.completions.create(
        model=section["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=section["temperature"],
        max_tokens=section["max_tokens"],
    )
    dt = time.perf_counter() - t0
    u = r.usage      # те, що звітує провайдер (для reasoning-моделей це включає приховані токени)
    return {"text": r.choices[0].message.content or "", "latency": dt,
            "tin": u.prompt_tokens if u else 0, "tout": u.completion_tokens if u else 0}


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
        return {f"{prefix}_calls": len(self.lat), f"{prefix}_tokens_in": self.tin,
                f"{prefix}_tokens_out": self.tout, f"{prefix}_cost_usd": cost,
                f"{prefix}_latency_p50_s": float(p50), f"{prefix}_latency_p95_s": float(p95),
                f"{prefix}_latency_first_call_s": self.lat[0],   # перший виклик: холодний кеш
                f"{prefix}_compute_seconds": float(sum(self.lat))}