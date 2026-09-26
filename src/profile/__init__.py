"""Distill normalized/ into profile/. See docs/plans/03-distill-profile.md."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NORMALIZED = ROOT / "normalized"
PROFILE_DIR = ROOT / "profile"
CACHE_DIR = PROFILE_DIR / "cache"
SPEND_PATH = PROFILE_DIR / "spend.jsonl"
LEDGER_PATH = PROFILE_DIR / "ledger.jsonl"
SELECTION_PATH = PROFILE_DIR / "selection.jsonl"
REDUCED_PATH = PROFILE_DIR / "reduced.jsonl"

MODEL = "deepseek/deepseek-v4.1-flash"
ACCOUNTS = ("tjbryant", "roadpiratefilms")

CATEGORIES = (
    "identity", "communication_style", "preference", "technical", "workflow",
    "project", "goal", "interest", "belief", "relationship",
    "health", "finance", "location", "surveillance_concern", "other",
)
# Categories dropped from every rendered file (kept in the ledger).
REDACT: frozenset[str] = frozenset({"surveillance_concern", "finance", "location"})
# Topic backstop: claims matching this are dropped from render whatever their category,
# and any rendered file matching it fails the run.
REDACT_PATTERN = re.compile(
    r"\b(being|been|is|was|am|get(?:ting)?) (tracked|followed|watched|monitored|surveilled|spied on)\b"
    r"|\bsurveil\w*|\bstalk\w*|\bdirected[- ]energy\b|\btracking (him|me|them|the user)\b"
    r"|\btracked (inside|in|at|from)\b",
    re.IGNORECASE,
)
# PII filter: claims matching are dropped from render, artifact lines matching are replaced
# with "[redacted]", and rendered files matching fail the run.
PII_PATTERN = re.compile(
    r"[\w.+-]+@[\w-]+(\.[\w-]+)+"                        # email addresses
    r"|\b(born|birth\w*|date of birth|dob)\b"            # birth date mentions
    r"|\b(age|aged) \d{1,3}\b|\b\d{1,3}[- ]years?[- ]old\b",
    re.IGNORECASE,
)
# Personal literals (city, birth date, ...): one per line, case-insensitive, kept out of git.
REDACT_TERMS_PATH = PROFILE_DIR / "redact_terms.txt"


def redact_terms() -> re.Pattern | None:
    if not REDACT_TERMS_PATH.exists():
        return None
    terms = [t.strip() for t in REDACT_TERMS_PATH.read_text().splitlines()
             if t.strip() and not t.startswith("#")]
    return re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE) if terms else None


def is_redacted(text: str) -> re.Match | None:
    """First match of any render-time redaction rule, or None."""
    terms = redact_terms()
    return (REDACT_PATTERN.search(text) or PII_PATTERN.search(text)
            or (terms.search(text) if terms else None))

SEED_CONFIDENCE = 0.9
ASR_PENALTY = 0.7         # sesame: ASR transcripts and memory
VOICE_MODE_PENALTY = 0.8  # meta.voice_mode conversations
