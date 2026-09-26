"""Stage 2: score conversations for personalization signal."""
from __future__ import annotations

import math
import re

import orjson

from src.profile import NORMALIZED

CORRECTION = re.compile(
    r"\b(no,? actually|i already (told|said)|i prefer|that'?s not what i|i said|"
    r"don'?t (do|use|say)|stop (doing|using)|i meant|not what i asked|i want you to|i'?d rather)\b",
    re.IGNORECASE,
)


def score(conv: dict) -> float:
    user = [m["text"] or "" for m in conv["messages"] if m["role"] == "user"]
    if not user:
        return 0.0
    corrections = sum(len(CORRECTION.findall(t)) for t in user)
    chars = sum(len(t) for t in user)
    return round(len(user) + 5 * corrections + math.log1p(chars), 3)


def selection() -> list[dict]:
    """[{uid, score}] sorted by score desc, uid asc. Conversations with no user turns score 0."""
    rows = []
    with (NORMALIZED / "conversations.jsonl").open("rb") as f:
        for raw in f:
            conv = orjson.loads(raw)
            rows.append({"uid": conv["uid"], "score": score(conv)})
    rows.sort(key=lambda r: (-r["score"], r["uid"]))
    return rows
