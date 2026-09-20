"""Adapter contract. See docs/SCHEMA.md."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Protocol, TypedDict


class Message(TypedDict):
    i: int
    role: str  # user|assistant|system
    text: str
    ts: str | None
    model: str | None


class Conversation(TypedDict):
    uid: str
    platform: str
    account: str
    native_id: str
    kind: str  # chat|design_chat|voice_call
    title: str | None
    created_at: str | None
    updated_at: str | None
    model: str | None
    project_ref: str | None
    messages: list[Message]
    meta: dict[str, Any]


class Artifact(TypedDict):
    uid: str
    platform: str
    account: str
    kind: str  # memory|reflection|project_instructions|project_doc|profile|custom_instructions
    source_path: str
    title: str | None
    created_at: str | None
    text: str
    meta: dict[str, Any]


class Adapter(Protocol):
    platform: str

    def discover(self, root: Path) -> Iterator[tuple[Path, str]]:
        """Yield (path, account) pairs to parse. No content read here."""
        ...

    def conversations(self, path: Path, account: str) -> Iterator[Conversation]:
        ...

    def artifacts(self, path: Path, account: str) -> Iterator[Artifact]:
        ...
