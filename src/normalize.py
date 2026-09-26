"""Run adapters into normalized/*.jsonl. Usage: python -m src.normalize [platform ...]"""
from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import orjson

from src.adapters import get_adapter
from src.exclusions import count_excluded, load_exclusions
from src.verify import EXPORTS_ROOT, VALID_ROLES, _source_conversation_count

OUT_DIR = Path(__file__).resolve().parent.parent / "normalized"

CONVERSATION_KINDS = {"chat", "design_chat", "voice_call"}
ARTIFACT_KINDS = {"memory", "reflection", "project_instructions", "project_doc", "profile", "custom_instructions"}


class CheckFailed(Exception):
    pass


def _has_export_files(platform_dir: Path) -> bool:
    return platform_dir.is_dir() and any(p.is_file() for p in platform_dir.rglob("*"))


def _check_ts(ts: str | None, where: str) -> datetime | None:
    if ts is None:
        return None
    if not isinstance(ts, str) or not ts.endswith("Z"):
        raise CheckFailed(f"{where}: timestamp not UTC Z: {ts!r}")
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _account_stats(platform: str, account: str, exported_dir: Path, adapter, exclusions,
                   seen_uids: set[str], conv_out, art_out) -> dict:
    conversations = 0
    messages = 0
    roles: Counter[str] = Counter()
    voice_mode = 0
    unreachable = 0
    dropped: Counter[str] = Counter()
    lo: datetime | None = None
    hi: datetime | None = None

    def observe(dt: datetime | None):
        nonlocal lo, hi
        if dt is None:
            return
        lo = dt if lo is None or dt < lo else lo
        hi = dt if hi is None or dt > hi else hi

    def claim(uid: str):
        if uid in seen_uids:
            raise CheckFailed(f"duplicate uid: {uid}")
        seen_uids.add(uid)

    for conv in adapter.conversations(exported_dir, account):
        uid = conv["uid"]
        claim(uid)
        if conv["kind"] not in CONVERSATION_KINDS:
            raise CheckFailed(f"{uid}: conversation kind {conv['kind']!r}")
        observe(_check_ts(conv["created_at"], f"{uid} created_at"))
        observe(_check_ts(conv["updated_at"], f"{uid} updated_at"))
        for m in conv["messages"]:
            if m["role"] not in VALID_ROLES:
                raise CheckFailed(f"{uid} msg {m['i']}: role {m['role']!r}")
            roles[m["role"]] += 1
            observe(_check_ts(m["ts"], f"{uid} msg {m['i']} ts"))
        conversations += 1
        messages += len(conv["messages"])
        voice_mode += bool(conv["meta"].get("voice_mode"))
        unreachable += conv["meta"].get("unreachable_node_count", 0)
        for btype, n in conv["meta"].get("dropped_content_blocks", {}).items():
            dropped[str(btype)] += n
        conv_out.write(orjson.dumps(conv) + b"\n")

    artifacts: Counter[str] = Counter()
    for art in adapter.artifacts(exported_dir, account):
        uid = art["uid"]
        claim(uid)
        if art["kind"] not in ARTIFACT_KINDS:
            raise CheckFailed(f"{uid}: artifact kind {art['kind']!r}")
        _check_ts(art["created_at"], f"{uid} created_at")
        artifacts[art["kind"]] += 1
        art_out.write(orjson.dumps(art) + b"\n")

    excluded = count_excluded(exclusions, platform, account)
    expected = _source_conversation_count(platform, exported_dir) - excluded
    if conversations != expected:
        raise CheckFailed(
            f"{platform}/{account}: conversations={conversations}, expected source-excluded={expected}"
        )

    return {
        "conversations": conversations,
        "messages": messages,
        "roles": dict(roles),
        "artifacts": dict(artifacts),
        "excluded": excluded,
        "voice_mode": voice_mode,
        "unreachable_nodes": unreachable,
        "dropped_blocks": dict(dropped),
        "date_range": [lo.isoformat().replace("+00:00", "Z") if lo else None,
                       hi.isoformat().replace("+00:00", "Z") if hi else None],
    }


def _totals(stats: dict) -> dict:
    total = {"conversations": 0, "messages": 0, "roles": Counter(), "artifacts": Counter(),
             "excluded": 0, "voice_mode": 0, "unreachable_nodes": 0, "dropped_blocks": Counter()}
    for accounts in stats.values():
        for s in accounts.values():
            for k in ("conversations", "messages", "excluded", "voice_mode", "unreachable_nodes"):
                total[k] += s[k]
            for k in ("roles", "artifacts", "dropped_blocks"):
                total[k].update(s[k])
    return {k: dict(v) if isinstance(v, Counter) else v for k, v in total.items()}


def run(platforms: list[str]) -> None:
    exclusions = load_exclusions()
    OUT_DIR.mkdir(exist_ok=True)
    finals = [OUT_DIR / n for n in ("conversations.jsonl", "artifacts.jsonl", "stats.json")]
    tmps = [p.with_name(p.name + ".tmp") for p in finals]

    stats: dict[str, dict] = {}
    seen_uids: set[str] = set()
    try:
        with tmps[0].open("wb") as conv_out, tmps[1].open("wb") as art_out:
            for platform in platforms:
                if not _has_export_files(EXPORTS_ROOT / platform):
                    print(f"skip {platform}: no export files")
                    continue
                adapter = get_adapter(platform)
                stats[platform] = {}
                for exported_dir, account in adapter.discover(EXPORTS_ROOT / platform):
                    s = _account_stats(platform, account, exported_dir, adapter, exclusions,
                                       seen_uids, conv_out, art_out)
                    stats[platform][account] = s
                    print(f"{platform}/{account}: {s['conversations']} conversations, "
                          f"{s['messages']} messages, {sum(s['artifacts'].values())} artifacts, "
                          f"excluded {s['excluded']}")
        tmps[2].write_bytes(orjson.dumps({"platforms": stats, "totals": _totals(stats)},
                                         option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    except BaseException:
        for t in tmps:
            t.unlink(missing_ok=True)
        raise

    for t, f in zip(tmps, finals):
        t.replace(f)
    print(f"wrote {', '.join(str(f.relative_to(OUT_DIR.parent)) for f in finals)}")


def main(argv: list[str]) -> None:
    if argv:
        platforms = argv
    else:
        platforms = sorted(p.name for p in EXPORTS_ROOT.iterdir() if p.is_dir())
    try:
        run(platforms)
    except CheckFailed as e:
        print(f"CHECK FAILED: {e}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
