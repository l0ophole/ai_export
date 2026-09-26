"""Adapter registry."""
from __future__ import annotations

import importlib

_REGISTRY = {
    "claude": ("src.adapters.claude", "ClaudeAdapter"),
    "grok": ("src.adapters.grok", "GrokAdapter"),
    "deepseek": ("src.adapters.deepseek", "DeepseekAdapter"),
    "sesame": ("src.adapters.sesame", "SesameAdapter"),
    "copilot": ("src.adapters.copilot", "CopilotAdapter"),
    "gemini": ("src.adapters.gemini", "GeminiAdapter"),
}

PLATFORMS = tuple(_REGISTRY)


def get_adapter(platform: str):
    if platform not in _REGISTRY:
        raise NotImplementedError(platform)
    module, cls = _REGISTRY[platform]
    return getattr(importlib.import_module(module), cls)()
