"""Definition-of-done checks for an adapter. Usage: python -m src.verify <platform>"""
from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

EXPORTS_ROOT = Path(__file__).resolve().parent.parent / "exports"

VALID_ROLES = {"user", "assistant", "system"}


def _source_conversation_count(platform: str, exported_dir: Path) -> int:
    if platform == "claude":
        import ijson

        n = 0
        conv_json = exported_dir / "conversations.json"
        if conv_json.exists():
            with conv_json.open("rb") as f:
                for _ in ijson.items(f, "item"):
                    n += 1
        design_dir = exported_dir / "design_chats"
        if design_dir.is_dir():
            n += len(list(design_dir.glob("*.json")))
        return n
    if platform == "grok":
        import ijson

        n = 0
        with exported_dir.open("rb") as f:
            for _ in ijson.items(f, "conversations.item"):
                n += 1
        return n
    if platform == "deepseek":
        import ijson

        n = 0
        with (exported_dir / "conversations.json").open("rb") as f:
            for _ in ijson.items(f, "item"):
                n += 1
        return n
    if platform == "sesame":
        from src.adapters.sesame import SesameAdapter

        eligible, _skipped = SesameAdapter()._eligible_calls(exported_dir)
        return len(eligible)
    if platform == "copilot":
        from src.adapters.copilot import CopilotAdapter

        return sum(1 for _ in CopilotAdapter().grouped_conversations(exported_dir))
    if platform == "gemini":
        from src.adapters.gemini import GeminiAdapter

        return sum(1 for _ in GeminiAdapter().grouped_conversations(exported_dir))
    raise NotImplementedError(platform)


def _sesame_diagnostics(exported_dir: Path) -> dict:
    from src.adapters.sesame import SesameAdapter

    eligible, skipped = SesameAdapter()._eligible_calls(exported_dir)
    total_calls = len(eligible) + len(skipped)
    return {
        "total_calls": total_calls,
        "eligible_calls": len(eligible),
        "skipped_calls": [
            {"call_number": c["call_number"], "reason": c["skip_reason"]} for c in skipped
        ],
    }


def _get_adapter(platform: str):
    if platform == "claude":
        from src.adapters.claude import ClaudeAdapter

        return ClaudeAdapter()
    if platform == "grok":
        from src.adapters.grok import GrokAdapter

        return GrokAdapter()
    if platform == "deepseek":
        from src.adapters.deepseek import DeepseekAdapter

        return DeepseekAdapter()
    if platform == "sesame":
        from src.adapters.sesame import SesameAdapter

        return SesameAdapter()
    if platform == "copilot":
        from src.adapters.copilot import CopilotAdapter

        return CopilotAdapter()
    if platform == "gemini":
        from src.adapters.gemini import GeminiAdapter

        return GeminiAdapter()
    raise NotImplementedError(platform)


def _copilot_diagnostics(exported_dir: Path) -> dict:
    from src.adapters.copilot import CopilotAdapter

    groups = list(CopilotAdapter().grouped_conversations(exported_dir))
    splits = sum(1 for _title, _rows, split_index in groups if split_index > 0)
    distinct_titles = len({title for title, _rows, _idx in groups})
    return {"distinct_titles": distinct_titles, "resulting_conversations": len(groups), "splits": splits}


def _gemini_diagnostics(exported_dir: Path) -> dict:
    from src.adapters.gemini import GeminiAdapter

    adapter = GeminiAdapter()
    groups, verb_counts, canvas_entries = adapter._canonical_groups(exported_dir)
    turn_groups = {c: es for c, es in groups.items() if any(e.verb in ("Prompted", "Branched") for e in es)}
    merged_groups = sum(1 for es in groups.values() if len({id_ for e in es for id_ in e.details_ids}) > 1)
    return {
        "verb_counts": dict(verb_counts),
        "canonical_conversation_groups": len(turn_groups),
        "groups_with_aliased_ids": merged_groups,
        "canvas_activity_entries": len(canvas_entries),
    }


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def run(platform: str) -> None:
    adapter = _get_adapter(platform)

    now = datetime.now(timezone.utc)
    rows = []

    for exported_dir, account in adapter.discover(EXPORTS_ROOT / platform):
        source_count = _source_conversation_count(platform, exported_dir)

        conv_count = 0
        msg_count = 0
        role_counts: Counter[str] = Counter()
        null_or_unknown_role = 0
        unreachable_total = 0
        dropped_blocks: Counter[str] = Counter()
        segment_merge_total = 0
        segment_mismatch_total = 0
        min_date: datetime | None = None
        max_date: datetime | None = None
        epoch_1970 = 0
        future_dates = 0

        def observe(dt: datetime | None):
            nonlocal min_date, max_date, epoch_1970, future_dates
            if dt is None:
                return
            if dt.year == 1970:
                epoch_1970 += 1
            if dt > now:
                future_dates += 1
            if min_date is None or dt < min_date:
                min_date = dt
            if max_date is None or dt > max_date:
                max_date = dt

        for conv in adapter.conversations(exported_dir, account):
            conv_count += 1
            observe(_parse_ts(conv.get("created_at")))
            observe(_parse_ts(conv.get("updated_at")))
            unreachable_total += conv["meta"].get("unreachable_node_count", 0)
            for btype, cnt in conv["meta"].get("dropped_content_blocks", {}).items():
                dropped_blocks[btype] += cnt
            segment_merge_total += conv["meta"].get("segment_merge_count", 0)
            segment_mismatch_total += abs(conv["meta"].get("segment_count_mismatch", 0))

            for m in conv["messages"]:
                msg_count += 1
                role = m["role"]
                if role not in VALID_ROLES:
                    null_or_unknown_role += 1
                else:
                    role_counts[role] += 1
                observe(_parse_ts(m.get("ts")))

        artifact_count = 0
        artifact_kinds: Counter[str] = Counter()
        for art in adapter.artifacts(exported_dir, account):
            artifact_count += 1
            artifact_kinds[art["kind"]] += 1

        rows.append(
            {
                "account": account,
                "source_count": source_count,
                "conv_count": conv_count,
                "msg_count": msg_count,
                "role_counts": dict(role_counts),
                "null_or_unknown_role": null_or_unknown_role,
                "unreachable_total": unreachable_total,
                "dropped_blocks": dict(dropped_blocks),
                "date_range": (min_date, max_date),
                "epoch_1970": epoch_1970,
                "future_dates": future_dates,
                "artifact_count": artifact_count,
                "artifact_kinds": dict(artifact_kinds),
                "segment_merge_total": segment_merge_total,
                "segment_mismatch_total": segment_mismatch_total,
                "sesame_diagnostics": _sesame_diagnostics(exported_dir) if platform == "sesame" else None,
                "copilot_diagnostics": _copilot_diagnostics(exported_dir) if platform == "copilot" else None,
                "gemini_diagnostics": _gemini_diagnostics(exported_dir) if platform == "gemini" else None,
            }
        )

        assert conv_count == source_count, (
            f"{account}: conversation count mismatch "
            f"(adapter={conv_count}, source={source_count})"
        )
        assert null_or_unknown_role == 0, f"{account}: {null_or_unknown_role} bad-role messages"
        assert epoch_1970 == 0, f"{account}: {epoch_1970} dates at epoch 1970"
        assert future_dates == 0, f"{account}: {future_dates} future dates"

    _print_table(platform, rows)


def _print_table(platform: str, rows: list[dict]) -> None:
    print(f"\n=== {platform} verify ===\n")
    for r in rows:
        lo, hi = r["date_range"]
        print(f"account: {r['account']}")
        print(f"  conversations: {r['conv_count']} (source: {r['source_count']}) {'OK' if r['conv_count'] == r['source_count'] else 'MISMATCH'}")
        print(f"  messages: {r['msg_count']}  roles: {r['role_counts']}  bad_role: {r['null_or_unknown_role']}")
        print(f"  date range: {lo.isoformat() if lo else None} .. {hi.isoformat() if hi else None}")
        print(f"  epoch_1970: {r['epoch_1970']}  future_dates: {r['future_dates']}")
        print(f"  unreachable tree nodes: {r['unreachable_total']}")
        print(f"  dropped content blocks: {r['dropped_blocks']}")
        print(f"  artifacts: {r['artifact_count']}  by kind: {r['artifact_kinds']}")
        if r.get("sesame_diagnostics") is not None:
            diag = r["sesame_diagnostics"]
            print(f"  segment merges (segments folded into a prior turn): {r['segment_merge_total']}")
            print(f"  total_segments-vs-len(conversation) mismatches: {r['segment_mismatch_total']}")
            print(f"  calls in calls_data.json: {diag['total_calls']}  eligible (parsed): {diag['eligible_calls']}")
            print(f"  calls skipped: {diag['skipped_calls']}")
        if r.get("copilot_diagnostics") is not None:
            diag = r["copilot_diagnostics"]
            print(f"  distinct titles: {diag['distinct_titles']}  title-collision splits: {diag['splits']}")
            print(f"  NOTE: source_count is derived from the same title+gap-split logic as the")
            print(f"        adapter (no independent native conversation-id exists for this platform).")
        if r.get("gemini_diagnostics") is not None:
            diag = r["gemini_diagnostics"]
            print(f"  activity entry verbs: {diag['verb_counts']}")
            print(f"  canonical conversation groups: {diag['canonical_conversation_groups']}  "
                  f"(groups with aliased/re-keyed ids: {diag['groups_with_aliased_ids']})")
            print(f"  canvas-creation activity entries (-> artifacts): {diag['canvas_activity_entries']}")
            print(f"  NOTE: source_count is derived from the same Details-link union-find grouping as")
            print(f"        the adapter (no separate native conversation manifest exists to cross-check).")
        print()

    total_conv = sum(r["conv_count"] for r in rows)
    total_msg = sum(r["msg_count"] for r in rows)
    total_art = sum(r["artifact_count"] for r in rows)
    print(f"TOTAL conversations: {total_conv}  messages: {total_msg}  artifacts: {total_art}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m src.verify <platform>", file=sys.stderr)
        raise SystemExit(1)
    run(sys.argv[1])
