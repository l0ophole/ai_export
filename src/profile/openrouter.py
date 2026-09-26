"""Minimal OpenRouter chat client: cache, retries, spend log. Never logs the key or bodies."""
from __future__ import annotations

import hashlib
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import orjson

from src.profile import CACHE_DIR, SPEND_PATH

API_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 6


class OpenRouterError(Exception):
    pass


def model_prices(model: str) -> list[tuple[float, float]]:
    """[(USD per prompt token, USD per completion token)] for every provider endpoint.
    Free GET. Which endpoint serves a call depends on routing, so estimates use the range."""
    with urllib.request.urlopen(f"https://openrouter.ai/api/v1/models/{model}/endpoints", timeout=60) as resp:
        endpoints = orjson.loads(resp.read())["data"]["endpoints"]
    if not endpoints:
        raise OpenRouterError(f"no endpoints for {model}")
    return [(float(e["pricing"]["prompt"]), float(e["pricing"]["completion"])) for e in endpoints]


def _cache_key(payload: dict) -> str:
    blob = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(payload["model"].encode() + blob).hexdigest()


def _full(payload: dict) -> dict:
    return {**payload, "provider": {"data_collection": "deny"}, "usage": {"include": True}}


def is_cached(payload: dict) -> bool:
    return (CACHE_DIR / f"{_cache_key(_full(payload))}.json").exists()


def chat(payload: dict, stage: str) -> dict:
    """POST a chat completion. Cached by sha256(model+payload); cache hits cost nothing."""
    payload = _full(payload)
    key = _cache_key(payload)
    cache_path = CACHE_DIR / f"{key}.json"
    if cache_path.exists():
        return orjson.loads(cache_path.read_bytes())

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise OpenRouterError("OPENROUTER_API_KEY not set")
    req = urllib.request.Request(
        API_URL,
        data=orjson.dumps(payload),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                body = orjson.loads(resp.read())
            break
        except urllib.error.HTTPError as e:
            if e.code not in RETRY_STATUS or attempt == MAX_ATTEMPTS - 1:
                raise OpenRouterError(f"{stage}: HTTP {e.code}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise OpenRouterError(f"{stage}: {type(e).__name__}") from None
        time.sleep(min(60, 2 ** attempt))
    if "error" in body or not body.get("choices"):
        raise OpenRouterError(f"{stage}: error response {body.get('error', {}).get('code')}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(orjson.dumps(body))
    usage = body.get("usage") or {}
    with SPEND_PATH.open("ab") as f:
        f.write(orjson.dumps({
            "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "stage": stage,
            "model": body.get("model", payload["model"]),
            "provider": body.get("provider"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "cost": usage.get("cost"),
            "cache_key": key,
        }) + b"\n")
    return body


def content_json(body: dict) -> dict | None:
    """Parse the assistant message as a JSON object; None if it isn't one. Never repaired."""
    text = body["choices"][0]["message"].get("content") or ""
    try:
        obj = orjson.loads(text)
    except orjson.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None
