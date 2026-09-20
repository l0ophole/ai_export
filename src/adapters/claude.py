"""Claude adapter. See docs/SCHEMA.md (claude section) for source shapes."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import ijson
import orjson

from src.adapter import Artifact, Conversation, Message

ROOT_SENTINEL = "00000000-0000-4000-8000-000000000000"

# Content-block types that carry displayable prose in chat_messages.content[].
_TEXT_BLOCK_TYPES = {"text"}


def _load_small_json(path: Path) -> Any:
    """For files known to be single-digit MB at most (never conversations.json)."""
    return orjson.loads(path.read_bytes())


def _stream_top_level_array(path: Path) -> Iterator[Any]:
    with path.open("rb") as f:
        yield from ijson.items(f, "item")


class ClaudeAdapter:
    platform = "claude"

    def discover(self, root: Path) -> Iterator[tuple[Path, str]]:
        for account_dir in sorted(root.iterdir()):
            exported = account_dir / "exported"
            if exported.is_dir():
                yield exported, account_dir.name

    # ---------------------------------------------------------------- conversations

    def conversations(self, path: Path, account: str) -> Iterator[Conversation]:
        conv_json = path / "conversations.json"
        if conv_json.exists():
            for raw in _stream_top_level_array(conv_json):
                yield self._conversation_from_chat_messages(raw, account, conv_json)

        design_dir = path / "design_chats"
        if design_dir.is_dir():
            for f in sorted(design_dir.glob("*.json")):
                raw = _load_small_json(f)
                yield self._conversation_from_design_chat(raw, account, f)

    def _linearize(self, chat_messages: list[dict]) -> tuple[list[dict], int]:
        if not chat_messages:
            return [], 0
        by_uuid = {m["uuid"]: m for m in chat_messages}
        referenced_as_parent = {m["parent_message_uuid"] for m in chat_messages}
        leaves = [m for m in chat_messages if m["uuid"] not in referenced_as_parent]
        if not leaves:
            leaves = chat_messages
        leaf = max(leaves, key=lambda m: m["created_at"])

        path: list[dict] = []
        cur = leaf
        seen: set[str] = set()
        while cur is not None and cur["uuid"] not in seen:
            path.append(cur)
            seen.add(cur["uuid"])
            parent_uuid = cur["parent_message_uuid"]
            if parent_uuid == ROOT_SENTINEL:
                break
            cur = by_uuid.get(parent_uuid)
        path.reverse()

        unreachable = len(chat_messages) - len(path)
        return path, unreachable

    def _conversation_from_chat_messages(
        self, raw: dict, account: str, source_path: Path
    ) -> Conversation:
        chat_messages = raw["chat_messages"]
        ordered, unreachable = self._linearize(chat_messages)

        dropped_blocks: Counter[str] = Counter()
        messages: list[Message] = []
        for i, m in enumerate(ordered):
            role = {"human": "user", "assistant": "assistant"}.get(m["sender"])
            if role is None:
                raise ValueError(f"unknown sender {m['sender']!r} in {source_path}")

            blocks = m.get("content") or []
            if blocks:
                text_parts = []
                for b in blocks:
                    btype = b.get("type")
                    if btype in _TEXT_BLOCK_TYPES:
                        text_parts.append(b.get("text", ""))
                    else:
                        dropped_blocks[btype] += 1
                text = "\n\n".join(p for p in text_parts if p)
            else:
                text = m.get("text", "")

            messages.append(
                Message(i=i, role=role, text=text, ts=m.get("created_at"), model=None)
            )

        native_id = raw["uuid"]
        return Conversation(
            uid=f"claude:{account}:{native_id}",
            platform="claude",
            account=account,
            native_id=native_id,
            kind="chat",
            title=raw.get("name") or None,
            created_at=raw.get("created_at"),
            updated_at=raw.get("updated_at"),
            model=None,
            project_ref=None,
            messages=messages,
            meta={
                "summary": raw.get("summary"),
                "unreachable_node_count": unreachable,
                "dropped_content_blocks": dict(dropped_blocks),
                "source_path": str(source_path),
            },
        )

    def _conversation_from_design_chat(
        self, raw: dict, account: str, source_path: Path
    ) -> Conversation:
        native_id = raw["uuid"]
        project = raw.get("project")
        project_ref = f"claude:{account}:{project['uuid']}" if project else None

        dropped_blocks: Counter[str] = Counter()
        messages: list[Message] = []
        for i, m in enumerate(raw.get("messages") or []):
            role = m["role"]
            if role not in ("user", "assistant"):
                raise ValueError(f"unknown role {role!r} in {source_path}")

            content = m.get("content") or {}
            if role == "user":
                text = content.get("content", "") or ""
                ts = content.get("timestamp") or m.get("created_at")
            else:
                text_parts = []
                for b in content.get("contentBlocks") or []:
                    btype = b.get("type")
                    if btype == "text":
                        text_parts.append(b.get("text", ""))
                    else:
                        dropped_blocks[btype] += 1
                text = "\n\n".join(p for p in text_parts if p)
                ts = m.get("created_at")

            messages.append(Message(i=i, role=role, text=text, ts=ts, model=None))

        return Conversation(
            uid=f"claude:{account}:{native_id}",
            platform="claude",
            account=account,
            native_id=native_id,
            kind="design_chat",
            title=raw.get("title") or None,
            created_at=raw.get("created_at"),
            updated_at=raw.get("updated_at"),
            model=None,
            project_ref=project_ref,
            messages=messages,
            meta={
                "dropped_content_blocks": dict(dropped_blocks),
                "source_path": str(source_path),
            },
        )

    # ------------------------------------------------------------------- artifacts

    def artifacts(self, path: Path, account: str) -> Iterator[Artifact]:
        yield from self._project_artifacts(path, account)
        yield from self._memory_artifacts(path, account)
        yield from self._reflection_artifacts(path, account)
        yield from self._profile_artifacts(path, account)

    def _project_artifacts(self, path: Path, account: str) -> Iterator[Artifact]:
        projects_dir = path / "projects"
        if not projects_dir.is_dir():
            return
        for f in sorted(projects_dir.glob("*.json")):
            raw = _load_small_json(f)
            project_uuid = raw["uuid"]

            template = raw.get("prompt_template") or ""
            if template:
                yield Artifact(
                    uid=f"claude:{account}:project_instructions:{project_uuid}",
                    platform="claude",
                    account=account,
                    kind="project_instructions",
                    source_path=str(f),
                    title=raw.get("name"),
                    created_at=raw.get("created_at"),
                    text=template,
                    meta={
                        "project_uuid": project_uuid,
                        "description": raw.get("description"),
                        "is_private": raw.get("is_private"),
                        "is_starter_project": raw.get("is_starter_project"),
                    },
                )

            for doc in raw.get("docs") or []:
                yield Artifact(
                    uid=f"claude:{account}:project_doc:{doc['uuid']}",
                    platform="claude",
                    account=account,
                    kind="project_doc",
                    source_path=str(f),
                    title=doc.get("filename"),
                    created_at=doc.get("created_at"),
                    text=doc.get("content", ""),
                    meta={"project_uuid": project_uuid, "project_name": raw.get("name")},
                )

    def _memory_artifacts(self, path: Path, account: str) -> Iterator[Artifact]:
        memories_dir = path / "memories"
        if not memories_dir.is_dir():
            return
        for f in sorted(memories_dir.glob("*.json")):
            raw = _load_small_json(f)
            account_uuid = raw.get("account_uuid")

            conv_memory = raw.get("conversations_memory") or ""
            if conv_memory:
                yield Artifact(
                    uid=f"claude:{account}:memory:conversations_memory:{account_uuid}",
                    platform="claude",
                    account=account,
                    kind="memory",
                    source_path=str(f),
                    title="conversations_memory",
                    created_at=None,
                    text=conv_memory,
                    meta={"account_uuid": account_uuid},
                )

            for project_uuid, text in (raw.get("project_memories") or {}).items():
                yield Artifact(
                    uid=f"claude:{account}:memory:project:{project_uuid}",
                    platform="claude",
                    account=account,
                    kind="memory",
                    source_path=str(f),
                    title=f"project_memory:{project_uuid}",
                    created_at=None,
                    text=text or "",
                    meta={"account_uuid": account_uuid, "project_uuid": project_uuid},
                )

            for mf in raw.get("memory_files") or []:
                mf_path = mf.get("path", "")
                digest = hashlib.sha256(mf_path.encode()).hexdigest()[:16]
                yield Artifact(
                    uid=f"claude:{account}:memory:file:{digest}",
                    platform="claude",
                    account=account,
                    kind="memory",
                    source_path=str(f),
                    title=mf_path,
                    created_at=None,
                    text=mf.get("content", ""),
                    meta={"account_uuid": account_uuid, "memory_path": mf_path},
                )

    def _reflection_artifacts(self, path: Path, account: str) -> Iterator[Artifact]:
        reflections_dir = path / "reflections"
        if not reflections_dir.is_dir():
            return
        for f in sorted(reflections_dir.glob("*.json")):
            raw = _load_small_json(f)
            account_uuid = raw.get("account_uuid")

            for kind, items in (("feedback", raw.get("feedback") or []),
                                 ("reflections", raw.get("reflections") or [])):
                for idx, item in enumerate(items):
                    period = item.get("period") if isinstance(item, dict) else None
                    text = orjson.dumps(
                        item.get("content", item) if isinstance(item, dict) else item
                    ).decode()
                    yield Artifact(
                        uid=f"claude:{account}:reflection:{kind}:{idx}",
                        platform="claude",
                        account=account,
                        kind="reflection",
                        source_path=str(f),
                        title=f"{kind}:{period}" if period else kind,
                        created_at=None,
                        text=text,
                        meta={"account_uuid": account_uuid, "reflection_kind": kind, "period": period},
                    )

    def _profile_artifacts(self, path: Path, account: str) -> Iterator[Artifact]:
        users_json = path / "users.json"
        if not users_json.exists():
            return
        raw = _load_small_json(users_json)
        users = raw if isinstance(raw, list) else [raw]
        for user in users:
            user_uuid = user.get("uuid")
            text = orjson.dumps({k: v for k, v in user.items() if k != "uuid"}).decode()
            yield Artifact(
                uid=f"claude:{account}:profile:{user_uuid}",
                platform="claude",
                account=account,
                kind="profile",
                source_path=str(users_json),
                title=user.get("full_name"),
                created_at=None,
                text=text,
                meta=dict(user),
            )
