"""Deepseek adapter. See docs/SCHEMA.md (openai_tree / deepseek section).

Correction to SCHEMA.md: role is NOT inferred from `model` being null — model
is populated ("deepseek-reasoner") on every observed node regardless of
sender. Role is inferred from fragment `type` instead: REQUEST/FILE-only
fragments are a user turn, RESPONSE/THINK/SEARCH/TOOL_* fragments are an
assistant turn. A node with `fragments: []` (observed 3x, always a childless
node whose parent carried a REQUEST) is treated as an assistant turn with an
empty generation.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import ijson
import orjson

from src.adapter import Artifact, Conversation, Message
from src.adapters.openai_tree import linearize_mapping

_ASSISTANT_FRAGMENT_TYPES = {"RESPONSE", "THINK", "SEARCH", "TOOL_OPEN", "TOOL_SEARCH", "TOOL_FIND"}
_USER_FRAGMENT_TYPES = {"REQUEST", "FILE"}
_TEXT_FRAGMENT_TYPE = {"user": "REQUEST", "assistant": "RESPONSE"}


def _to_utc_z(ts: str) -> str:
    from datetime import timezone

    return datetime.fromisoformat(ts).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_small_json(path: Path) -> Any:
    return orjson.loads(path.read_bytes())


def _infer_role(fragments: list[dict], source_path: Path, native_id: str, node_id: str) -> str:
    types = {f["type"] for f in fragments}
    if types & _ASSISTANT_FRAGMENT_TYPES:
        return "assistant"
    if types & _USER_FRAGMENT_TYPES:
        return "user"
    if not fragments:
        return "assistant"
    raise ValueError(
        f"unrecognized fragment types {types!r} in {source_path} conv={native_id} node={node_id}"
    )


class DeepseekAdapter:
    platform = "deepseek"

    def discover(self, root: Path) -> Iterator[tuple[Path, str]]:
        for account_dir in sorted(root.iterdir()):
            exported = account_dir / "exported"
            if (exported / "conversations.json").exists():
                yield exported, account_dir.name

    # ---------------------------------------------------------------- conversations

    def conversations(self, path: Path, account: str) -> Iterator[Conversation]:
        conv_json = path / "conversations.json"
        with conv_json.open("rb") as f:
            for raw in ijson.items(f, "item"):
                yield self._conversation(raw, account, conv_json)

    def _conversation(self, raw: dict, account: str, source_path: Path) -> Conversation:
        native_id = raw["id"]
        mapping = raw["mapping"]
        ordered, unreachable = linearize_mapping(mapping)

        dropped: Counter[str] = Counter()
        think_by_index: dict[str, str] = {}
        messages: list[Message] = []
        for i, node in enumerate(ordered):
            msg = node["message"]
            fragments = msg["fragments"]
            role = _infer_role(fragments, source_path, native_id, node["id"])

            text_parts = []
            think_parts = []
            wanted = _TEXT_FRAGMENT_TYPE[role]
            for frag in fragments:
                ftype = frag["type"]
                if ftype == wanted:
                    text_parts.append(frag.get("content", ""))
                elif ftype == "THINK":
                    think_parts.append(frag.get("content", ""))
                else:
                    dropped[ftype] += 1
            text = "\n\n".join(p for p in text_parts if p)
            if think_parts:
                think_by_index[str(i)] = "\n\n".join(p for p in think_parts if p)

            messages.append(
                Message(
                    i=i,
                    role=role,
                    text=text,
                    ts=_to_utc_z(msg["inserted_at"]),
                    model=msg.get("model"),
                )
            )

        return Conversation(
            uid=f"deepseek:{account}:{native_id}",
            platform="deepseek",
            account=account,
            native_id=native_id,
            kind="chat",
            title=raw.get("title") or None,
            created_at=_to_utc_z(raw["inserted_at"]),
            updated_at=_to_utc_z(raw["updated_at"]),
            model=None,
            project_ref=None,
            messages=messages,
            meta={
                "unreachable_node_count": unreachable,
                "dropped_content_blocks": dict(dropped),
                "think_by_index": think_by_index,
                "source_path": str(source_path),
            },
        )

    # ------------------------------------------------------------------- artifacts

    def artifacts(self, path: Path, account: str) -> Iterator[Artifact]:
        user_json = path / "user.json"
        if not user_json.exists():
            return
        raw = _load_small_json(user_json)
        user_id = raw.get("user_id")
        text = orjson.dumps({k: v for k, v in raw.items() if k != "user_id"}).decode()
        yield Artifact(
            uid=f"deepseek:{account}:profile:{user_id}",
            platform="deepseek",
            account=account,
            kind="profile",
            source_path=str(user_json),
            title=None,
            created_at=None,
            text=text,
            meta=dict(raw),
        )
