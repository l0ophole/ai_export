"""Gemini adapter. See docs/SCHEMA.md gemini section -- STALE, see proposal
delivered alongside this file (SCHEMA.md still says "Blocked"/JSON; the real
export is Takeout `MyActivity.html`, confirmed present for both accounts).

Source: `exports/gemini/<account>/MyActivity.html`, a Google Takeout activity
log. One `<div class="outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp">`
per activity entry, newest-first. Per-entry body starts with a verb:

- "Prompted <text>" / "Branched <text>" -- a real turn. The assistant's reply
  is embedded in the SAME entry, after the timestamp -- these are pre-paired
  turns, not a raw event stream. Both verbs carry a `Details:` block with one
  or more `https://gemini.google.com/app/<id>` links -- the real native
  conversation id, unlike copilot/sesame's lack of one.
- "Created Gemini Canvas titled <title>" -- a generated Canvas (HTML/code),
  no Details link, no thread membership. Routed to artifacts.jsonl
  (kind=project_doc; closest fit in the fixed enum, flagged in the proposal).
- "Used Gemini Apps" / "Cleared previous feedback" -- telemetry, no content.
  Counted, never emitted as messages or artifacts.

Multiple ids under one entry's Details block (10 confirmed, tjbryant) mean
Gemini re-keyed the conversation (edit/regenerate) -- union-find merges them
into one canonical group so a renamed thread doesn't split in two.

Timezone: every timestamp is suffixed "CDT" including entries in December/
January, when the account's real zone (America/Chicago) would be CST. This is
a known Takeout quirk -- it stamps the account's *current* offset abbreviation
on every historical row rather than the DST-correct one. Fix: parse the naive
local wall-clock value and localize with zoneinfo("America/Chicago"), which
applies real DST for that date; the literal "CDT" text is not trusted as an
offset. Fails loudly (NotImplementedError) if a non-CDT abbreviation ever
shows up, since that would mean this assumption doesn't hold for some row.
"""
from __future__ import annotations

import hashlib
import html
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

from src.adapter import Artifact, Conversation, Message

SOURCE_TZ = ZoneInfo("America/Chicago")

_ENTRY_MARKER = '<div class="outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp">'
_TS_RE = re.compile(
    r"([A-Z][a-z]{2} \d{1,2}, \d{4}, \d{1,2}:\d{2}:\d{2})\s*(AM|PM)\s*([A-Z]{2,4})"
)
_DETAILS_RE = re.compile(r"<b>Details:</b><br>(.*?)<b>", re.S)
_DETAILS_ID_RE = re.compile(r"gemini\.google\.com/app/([a-f0-9]+)")
_LOCAL_HREF_RE = re.compile(r'<a href="([^"]+)">')
_TAG_RE = re.compile(r"<[^>]+>")

_TURN_VERBS = ("Prompted", "Branched")
_CANVAS_VERB = "Created"
_TELEMETRY_VERBS = ("Used", "Cleared")


def _html_to_text(fragment: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", fragment)
    text = re.sub(r"</p>", "\n\n", text)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ").replace(" ", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_ts(date_str: str, ampm: str, tz_abbrev: str) -> datetime:
    if tz_abbrev != "CDT":
        raise NotImplementedError(f"unhandled gemini timestamp tz abbrev {tz_abbrev!r}")
    naive = datetime.strptime(f"{date_str} {ampm}", "%b %d, %Y, %I:%M:%S %p")
    return naive.replace(tzinfo=SOURCE_TZ)


def _to_utc_z(dt: datetime) -> str:
    return dt.astimezone(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)


class _Entry:
    __slots__ = ("verb", "before", "after", "ts", "details_ids", "attachments", "raw")

    def __init__(self, verb: str, before: str, after: str, ts: datetime, details_ids: list[str], attachments: list[str], raw: str):
        self.verb = verb
        self.before = before
        self.after = after
        self.ts = ts
        self.details_ids = details_ids
        self.attachments = attachments
        self.raw = raw


class GeminiAdapter:
    platform = "gemini"

    def discover(self, root: Path) -> Iterator[tuple[Path, str]]:
        for account_dir in sorted(root.iterdir()):
            if (account_dir / "MyActivity.html").exists():
                yield account_dir, account_dir.name

    # ---------------------------------------------------------------- parsing

    def _iter_entries(self, path: Path) -> Iterator[_Entry]:
        html_path = path / "MyActivity.html"
        content = html_path.read_text(encoding="utf-8")
        parts = content.split(_ENTRY_MARKER)[1:]
        for raw in parts:
            m = re.search(
                r'content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">(.*?)'
                r'</div><div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1 mdl-typography--text-right"',
                raw,
                re.S,
            )
            if not m:
                continue
            body = m.group(1)
            ts_match = _TS_RE.search(body)
            if not ts_match:
                continue  # no timestamp: not an activity row we can place in time
            ts = _parse_ts(*ts_match.groups())
            before = body[: ts_match.start()]
            after = body[ts_match.end():].lstrip()
            if after.startswith("<br>"):
                after = after[4:]

            verb = before.strip().split(" ", 1)[0].split("\xa0")[0]

            details_block = _DETAILS_RE.search(raw)
            details_ids = list(dict.fromkeys(_DETAILS_ID_RE.findall(details_block.group(1)))) if details_block else []
            attachments = [h for h in _LOCAL_HREF_RE.findall(before) if not h.startswith("http")]

            yield _Entry(verb, before, after, ts, details_ids, attachments, raw)

    def _canonical_groups(self, path: Path) -> tuple[dict[str, list[_Entry]], Counter, list[_Entry]]:
        """Returns (canonical_id -> entries with a turn or telemetry tied to it,
        verb_counts, canvas_entries)."""
        uf = _UnionFind()
        entries = list(self._iter_entries(path))
        for e in entries:
            for a, b in zip(e.details_ids, e.details_ids[1:]):
                uf.union(a, b)

        verb_counts: Counter[str] = Counter()
        canvas_entries: list[_Entry] = []
        groups: dict[str, list[_Entry]] = {}
        for e in entries:
            verb_counts[e.verb] += 1
            if e.verb == _CANVAS_VERB:
                canvas_entries.append(e)
                continue
            if e.verb in _TELEMETRY_VERBS and not e.details_ids:
                continue
            if not e.details_ids:
                continue
            canon = uf.find(e.details_ids[0])
            groups.setdefault(canon, []).append(e)
        return groups, verb_counts, canvas_entries

    def grouped_conversations(self, path: Path) -> Iterator[tuple[str, list[_Entry]]]:
        """Shared by conversations() and verify.py: canonical id -> its entries,
        turn-bearing entries only. No independent native conversation count
        exists for this platform either -- the Details-link id is real, but
        there is no separate manifest to cross-check the grouping against, so
        this generator IS the source of truth for both the adapter and the
        verify count (same situation as copilot)."""
        groups, _verb_counts, _canvas = self._canonical_groups(path)
        for canon, group_entries in groups.items():
            turn_entries = [e for e in group_entries if e.verb in _TURN_VERBS]
            if not turn_entries:
                continue
            yield canon, turn_entries

    def conversations(self, path: Path, account: str) -> Iterator[Conversation]:
        for canon, turn_entries in self.grouped_conversations(path):
            yield self._conversation(canon, turn_entries, account, path)

    def _conversation(self, canon: str, turn_entries: list[_Entry], account: str, path: Path) -> Conversation:
        ordered = sorted(turn_entries, key=lambda e: e.ts)
        messages: list[Message] = []
        attachments: list[str] = []
        i = 0
        for e in ordered:
            ts = _to_utc_z(e.ts)
            user_text = _html_to_text(re.sub(rf"^{re.escape(e.verb)}", "", e.before, count=1))
            messages.append(Message(i=i, role="user", text=user_text, ts=ts, model=None))
            i += 1
            messages.append(Message(i=i, role="assistant", text=_html_to_text(e.after), ts=ts, model=None))
            i += 1
            attachments.extend(e.attachments)

        all_ids = sorted({id_ for e in ordered for id_ in e.details_ids})
        return Conversation(
            uid=f"gemini:{account}:{canon}",
            platform="gemini",
            account=account,
            native_id=canon,
            kind="chat",
            title=None,
            created_at=_to_utc_z(ordered[0].ts),
            updated_at=_to_utc_z(ordered[-1].ts),
            model=None,
            project_ref=None,
            messages=messages,
            meta={
                "unreachable_node_count": 0,
                "dropped_content_blocks": {},
                "source_path": str(path / "MyActivity.html"),
                "aliased_conversation_ids": all_ids,
                "attachments": attachments,
                "verb_counts": dict(Counter(e.verb for e in ordered)),
            },
        )

    # ------------------------------------------------------------------- artifacts

    def artifacts(self, path: Path, account: str) -> Iterator[Artifact]:
        html_path = path / "MyActivity.html"
        _groups, verb_counts, canvas_entries = self._canonical_groups(path)

        referenced: set[str] = set()
        for e in self._iter_entries(path):
            referenced.update(e.attachments)

        for e in canvas_entries:
            title_match = re.match(r"Created Gemini Canvas titled\s*(.*?)<br>", e.before, re.S)
            if not title_match:
                raise ValueError(f"'Created' entry didn't match expected canvas shape: {e.before[:200]!r}")
            title = _html_to_text(title_match.group(1))
            body = e.before[title_match.end():]
            native_id = hashlib.sha256(f"{account}|canvas|{e.ts.isoformat()}|{title}".encode()).hexdigest()[:24]
            yield Artifact(
                uid=f"gemini:{account}:canvas:{native_id}",
                platform="gemini",
                account=account,
                kind="project_doc",
                source_path=str(html_path),
                title=title or None,
                created_at=_to_utc_z(e.ts),
                text=_html_to_text(body),
                meta={"verb": "Created", "note": "Canvas creation event from activity log, not the rendered Canvas app itself"},
            )

        canvas_files = sorted(path.glob("chara_canvas-*.html"))
        for cf in canvas_files:
            src = cf.read_text(encoding="utf-8")
            title_match = re.search(r"<title>(.*?)</title>", src)
            yield Artifact(
                uid=f"gemini:{account}:canvas_app:{cf.stem}",
                platform="gemini",
                account=account,
                kind="project_doc",
                source_path=str(cf),
                title=html.unescape(title_match.group(1)) if title_match else cf.name,
                created_at=None,
                text=src,
                meta={"note": "Standalone Gemini Canvas app export (self-contained HTML/JS), not an activity-log entry"},
            )

        loose_files = [
            p.name
            for p in path.iterdir()
            if p.is_file() and p.name != "MyActivity.html" and not p.name.startswith("chara_canvas-")
        ]
        orphaned = sorted(set(loose_files) - referenced)
        yield Artifact(
            uid=f"gemini:{account}:media_manifest",
            platform="gemini",
            account=account,
            kind="profile",
            source_path=str(path),
            title=None,
            created_at=None,
            text="",
            meta={
                "media_file_count": len(loose_files),
                "referenced_media_files": sorted(set(loose_files) & referenced),
                "orphaned_media_files": orphaned,
                "telemetry_entry_counts": {v: c for v, c in verb_counts.items() if v in _TELEMETRY_VERBS},
            },
        )
