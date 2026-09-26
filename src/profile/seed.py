"""Stage 1: artifacts -> ledger lines at high confidence. No LLM."""
from __future__ import annotations

import hashlib
from collections import Counter

import orjson

from src.profile import ASR_PENALTY, NORMALIZED, SEED_CONFIDENCE


def line_id(source: str, conv_uid: str, msg_i: list[int], claim: str) -> str:
    h = hashlib.sha256(orjson.dumps([source, conv_uid, msg_i, claim]))
    return h.hexdigest()[:16]


def seed_lines() -> tuple[list[dict], Counter]:
    """One ledger line per artifact with non-empty text. Claude conversation `summary`
    lives in conversation meta, never in artifacts.jsonl, so it cannot enter here."""
    lines: list[dict] = []
    counts: Counter[str] = Counter()
    with (NORMALIZED / "artifacts.jsonl").open("rb") as f:
        for raw in f:
            art = orjson.loads(raw)
            text = art["text"] or ""
            if not text.strip():
                counts["skipped_empty_text"] += 1
                continue
            confidence = SEED_CONFIDENCE
            if art["platform"] == "sesame":
                confidence *= ASR_PENALTY
            lines.append({
                "id": line_id("seed", art["uid"], [], text),
                "claim": text,
                "category": f"artifact:{art['kind']}",
                "confidence": round(confidence, 3),
                "account": art["account"],
                "conv_uid": art["uid"],
                "msg_i": [],
                "ts": art["created_at"],
                "source": "seed",
                "model": None,
                "title": art["title"],
            })
            counts[f"seeded:{art['kind']}"] += 1
    return lines, counts
