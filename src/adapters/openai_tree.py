"""Shared linearization for the openai_tree mapping shape (chatgpt + deepseek).

See docs/SCHEMA.md's "openai_tree" section for the shape and the
find-leaf/walk-parent/skip-null-message/report-unreachable rule.
"""
from __future__ import annotations

from typing import Any


def linearize_mapping(
    mapping: dict[str, dict[str, Any]], current_node: str | None = None
) -> tuple[list[dict[str, Any]], int]:
    nodes_with_message = [n for n in mapping.values() if n.get("message") is not None]
    if not nodes_with_message:
        return [], 0

    leaf = mapping.get(current_node) if current_node else None
    if leaf is None or leaf.get("message") is None:
        referenced_as_parent = {n["parent"] for n in mapping.values() if n.get("parent")}
        leaves = [n for n in nodes_with_message if n["id"] not in referenced_as_parent]
        if not leaves:
            leaves = nodes_with_message
        leaf = max(leaves, key=lambda n: n["message"]["inserted_at"])

    path: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = leaf
    seen: set[str] = set()
    while cur is not None and cur["id"] not in seen:
        if cur.get("message") is not None:
            path.append(cur)
        seen.add(cur["id"])
        parent_id = cur.get("parent")
        cur = mapping.get(parent_id) if parent_id else None
    path.reverse()

    unreachable = len(nodes_with_message) - len(path)
    return path, unreachable
