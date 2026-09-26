"""Conversation exclusions: repo-root exclusions.json, a {uid: reason} object. Gitignored."""
from __future__ import annotations

from pathlib import Path

import orjson

EXCLUSIONS_PATH = Path(__file__).resolve().parent.parent / "exclusions.json"


def load_exclusions(path: Path = EXCLUSIONS_PATH) -> dict[str, str]:
    if not path.exists():
        return {}
    data = orjson.loads(path.read_bytes())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a {{uid: reason}} object")
    return data


def count_excluded(exclusions: dict[str, str], platform: str, account: str) -> int:
    prefix = f"{platform}:{account}:"
    return sum(1 for uid in exclusions if uid.startswith(prefix))
