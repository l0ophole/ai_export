"""Copilot adapter. See docs/SCHEMA.md (copilot section) for source shapes.

Assumption beyond SCHEMA.md: `Time` carries no offset (e.g. `2026-09-18T00:48:04`).
Treated as America/Chicago local time (the account owner's zone) and converted
to UTC via zoneinfo (DST-aware) -- the corpus spans both DST and standard-time
months. If this assumption is wrong, timestamps are off by a fixed few hours,
not corrupted structurally.

Split threshold: 6 hours. Scanned the gap between consecutive distinct
timestamps within the same title across both accounts (93 gaps): 98th
percentile is 2.26h, then the next gap jumps to 62.3h -- 6h sits cleanly in
that empty range. Confirmed against a real title collision ("Enhanced Prompt
for Character Card Analysis Script", tjbryant) that reappears ~62h apart.

Rows are NOT strictly alternating Human/AI pairs at matching timestamps --
observed voice-call-derived turns and an inserted "I started the page, ..."
system-style AI message break that pattern. Messages are just sorted
ascending by `Time` per title, with `Human` ordered before `AI` on an exact
tie (same-second turn).
"""
from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterator
from zoneinfo import ZoneInfo

from src.adapter import Artifact, Conversation, Message

SPLIT_THRESHOLD_SECONDS = 6 * 3600
SOURCE_TZ = ZoneInfo("America/Chicago")

_ROLE_MAP = {"Human": "user", "AI": "assistant"}
_TIE_ORDER = {"Human": 0, "AI": 1}


def _to_utc_z(naive: datetime) -> str:
    return naive.replace(tzinfo=SOURCE_TZ).astimezone(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")


class CopilotAdapter:
    platform = "copilot"

    def discover(self, root: Path) -> Iterator[tuple[Path, str]]:
        for account_dir in sorted(root.iterdir()):
            for csv_path in sorted(account_dir.glob("copilot-*.csv")):
                yield csv_path, account_dir.name

    # ---------------------------------------------------------------- conversations

    def _grouped_rows(self, path: Path) -> dict[str, list[tuple[datetime, str, str]]]:
        by_title: dict[str, list[tuple[datetime, str, str]]] = defaultdict(list)
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["Conversation", "Time", "Author", "Message"], (
                f"unexpected header {header!r} in {path}"
            )
            for title, ts, author, message in reader:
                if author not in _ROLE_MAP:
                    raise ValueError(f"unknown Author {author!r} in {path}")
                by_title[title].append((datetime.fromisoformat(ts), author, message))
        for rows in by_title.values():
            rows.sort(key=lambda r: (r[0], _TIE_ORDER[r[1]]))
        return by_title

    def _split_groups(
        self, rows: list[tuple[datetime, str, str]]
    ) -> list[list[tuple[datetime, str, str]]]:
        groups: list[list[tuple[datetime, str, str]]] = [[rows[0]]]
        for prev, cur in zip(rows, rows[1:]):
            gap = (cur[0] - prev[0]).total_seconds()
            if gap > SPLIT_THRESHOLD_SECONDS:
                groups.append([])
            groups[-1].append(cur)
        return groups

    def grouped_conversations(
        self, path: Path
    ) -> Iterator[tuple[str, list[tuple[datetime, str, str]], int]]:
        """Yields (title, ordered_rows, split_index) per conversation group.

        Shared by conversations() and verify.py's diagnostics, since there's
        no independent native conversation count for this platform -- the
        "source count" is derived from this same title+gap logic, not an
        independent cross-check.
        """
        for title, rows in self._grouped_rows(path).items():
            for split_index, group in enumerate(self._split_groups(rows)):
                yield title, group, split_index

    def conversations(self, path: Path, account: str) -> Iterator[Conversation]:
        for title, group, split_index in self.grouped_conversations(path):
            yield self._conversation(title, group, split_index, account, path)

    def _conversation(
        self,
        title: str,
        group: list[tuple[datetime, str, str]],
        split_index: int,
        account: str,
        source_path: Path,
    ) -> Conversation:
        first_ts = group[0][0]
        native_id = hashlib.sha256(
            f"{title}|{first_ts.isoformat()}".encode()
        ).hexdigest()[:24]

        messages: list[Message] = []
        for i, (ts, author, text) in enumerate(group):
            messages.append(
                Message(i=i, role=_ROLE_MAP[author], text=text, ts=_to_utc_z(ts), model=None)
            )

        created_at = _to_utc_z(group[0][0])
        updated_at = _to_utc_z(group[-1][0])
        return Conversation(
            uid=f"copilot:{account}:{native_id}",
            platform="copilot",
            account=account,
            native_id=native_id,
            kind="chat",
            title=title,
            created_at=created_at,
            updated_at=updated_at,
            model=None,
            project_ref=None,
            messages=messages,
            meta={
                "unreachable_node_count": 0,
                "dropped_content_blocks": {},
                "title_split_index": split_index,
                "source_path": str(source_path),
            },
        )

    # ------------------------------------------------------------------- artifacts

    def artifacts(self, path: Path, account: str) -> Iterator[Artifact]:
        return iter(())  # no artifacts in this source: no memories/profile/instructions in the CSV
