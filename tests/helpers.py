"""Shared helpers for per-adapter fixture tests."""
from __future__ import annotations

import re
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
_UTC_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")


def run(adapter):
    """Discover over the platform's fixture root; return (accounts, conversations, artifacts)."""
    accounts, convs, arts = [], [], []
    for path, account in adapter.discover(FIXTURES / adapter.platform):
        accounts.append(account)
        convs.extend(adapter.conversations(path, account))
        arts.extend(adapter.artifacts(path, account))
    return accounts, convs, arts


def roles(conv):
    return [m["role"] for m in conv["messages"]]


def texts(conv):
    return [m["text"] for m in conv["messages"]]


def assert_utc_z(tc, ts):
    tc.assertIsNotNone(ts)
    tc.assertRegex(ts, _UTC_Z)


def assert_messages_utc_z(tc, conv):
    for m in conv["messages"]:
        assert_utc_z(tc, m["ts"])
    tc.assertEqual([m["i"] for m in conv["messages"]], list(range(len(conv["messages"]))))
