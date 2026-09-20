"""Grok adapter. See docs/SCHEMA.md (grok section) for source shapes."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import ijson
import orjson

from src.adapter import Artifact, Conversation, Message

_SENDER_MAP = {"human": "user", "assistant": "assistant", "ASSISTANT": "assistant"}

# Response fields that carry tool/reasoning metadata, not displayable text.
_DROPPED_FIELDS = (
    "agent_thinking_traces",
    "steps",
    "web_search_results",
    "card_attachments_json",
    "xpost_ids",
    "file_attachments",
    "thinking_trace",
)


def _mongo_date_to_iso(d: dict) -> str:
    ms = int(d["$date"]["$numberLong"])
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _load_small_json(path: Path) -> Any:
    return orjson.loads(path.read_bytes())


class GrokAdapter:
    platform = "grok"

    def discover(self, root: Path) -> Iterator[tuple[Path, str]]:
        for account_dir in sorted(root.iterdir()):
            for f in sorted(account_dir.glob("exported/ttl/*/export_data/*/prod-grok-backend.json")):
                yield f, account_dir.name

    # ---------------------------------------------------------------- conversations

    def _linearize(self, responses: list[dict], leaf_response_id: str | None) -> tuple[list[dict], int]:
        if not responses:
            return [], 0
        by_id = {r["_id"]: r for r in responses}

        leaf = by_id.get(leaf_response_id) if leaf_response_id else None
        if leaf is None:
            referenced_as_parent = {
                r["parent_response_id"] for r in responses if r.get("parent_response_id")
            }
            leaves = [r for r in responses if r["_id"] not in referenced_as_parent]
            if not leaves:
                leaves = responses
            leaf = max(leaves, key=lambda r: r["create_time"]["$date"]["$numberLong"])

        path: list[dict] = []
        cur = leaf
        seen: set[str] = set()
        while cur is not None and cur["_id"] not in seen:
            path.append(cur)
            seen.add(cur["_id"])
            parent_id = cur.get("parent_response_id")
            cur = by_id.get(parent_id) if parent_id else None
        path.reverse()

        unreachable = len(responses) - len(path)
        return path, unreachable

    def conversations(self, path: Path, account: str) -> Iterator[Conversation]:
        with path.open("rb") as f:
            for entry in ijson.items(f, "conversations.item"):
                yield self._conversation(entry, account, path)

    def _conversation(self, entry: dict, account: str, source_path: Path) -> Conversation:
        conv = entry["conversation"]
        responses = [r["response"] for r in entry.get("responses") or []]
        ordered, unreachable = self._linearize(responses, conv.get("leaf_response_id"))

        dropped: Counter[str] = Counter()
        messages: list[Message] = []
        for i, r in enumerate(ordered):
            role = _SENDER_MAP.get(r["sender"])
            if role is None:
                raise ValueError(f"unknown sender {r['sender']!r} in {source_path}")

            for field in _DROPPED_FIELDS:
                if r.get(field):
                    dropped[field] += 1

            messages.append(
                Message(
                    i=i,
                    role=role,
                    text=r.get("message") or "",
                    ts=_mongo_date_to_iso(r["create_time"]) if r.get("create_time") else None,
                    model=r.get("model") or None,
                )
            )

        native_id = conv["id"]
        return Conversation(
            uid=f"grok:{account}:{native_id}",
            platform="grok",
            account=account,
            native_id=native_id,
            kind="chat",
            title=conv.get("title") or None,
            created_at=conv.get("create_time"),
            updated_at=conv.get("modify_time"),
            model=None,
            project_ref=None,
            messages=messages,
            meta={
                "summary": conv.get("summary"),
                "unreachable_node_count": unreachable,
                "dropped_content_blocks": dict(dropped),
                "leaf_response_id_was_null": conv.get("leaf_response_id") is None,
                "system_prompt_name": conv.get("system_prompt_name") or None,
                "source_path": str(source_path),
            },
        )

    # ------------------------------------------------------------------- artifacts

    def artifacts(self, path: Path, account: str) -> Iterator[Artifact]:
        raw = _load_small_json(path)

        for project in raw.get("projects") or []:
            personality = project.get("custom_personality") or ""
            if personality:
                yield Artifact(
                    uid=f"grok:{account}:project_instructions:{project['workspace_id']}",
                    platform="grok",
                    account=account,
                    kind="project_instructions",
                    source_path=str(path),
                    title=project.get("name"),
                    created_at=project.get("create_time"),
                    text=personality,
                    meta={"workspace_id": project["workspace_id"]},
                )

        tasks = raw.get("tasks") or []
        if tasks:
            raise NotImplementedError(
                f"grok tasks are non-empty for {account} ({len(tasks)}) — "
                "shape unconfirmed, extend adapter before proceeding"
            )

        media_posts = raw.get("media_posts") or []
        if media_posts:
            yield Artifact(
                uid=f"grok:{account}:media_posts_index",
                platform="grok",
                account=account,
                kind="profile",
                source_path=str(path),
                title="media_posts",
                created_at=None,
                text="",
                meta={
                    "note": "image content skipped; id mapping recorded so nothing is orphaned",
                    "posts": [
                        {
                            "id": p.get("id"),
                            "media_type": p.get("media_type"),
                            "create_time": p.get("create_time"),
                            "link": p.get("link"),
                            "original_prompt": p.get("original_prompt"),
                        }
                        for p in media_posts
                    ],
                },
            )
