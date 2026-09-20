"""Sesame adapter. See docs/SCHEMA.md (sesame section) for source shapes."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import orjson

from src.adapter import Artifact, Conversation, Message

_ASR_CONFIDENCE_NOTE = (
    "Transcript produced by ASR (automatic speech recognition) on a live voice "
    "call. Misheard or misrecognized words can read as confidently stated facts."
)


def _load_small_json(path: Path) -> Any:
    return orjson.loads(path.read_bytes())


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


class SesameAdapter:
    platform = "sesame"

    def discover(self, root: Path) -> Iterator[tuple[Path, str]]:
        calls_data = root / "calls_data.json"
        if not calls_data.exists():
            return
        user_info_path = root / "user_info.json"
        account = "tjbryant"
        if user_info_path.exists():
            email = _load_small_json(user_info_path).get("email", "")
            if email:
                account = email.split("@")[0]
        yield root, account

    # ---------------------------------------------------------------- conversations

    def _eligible_calls(self, root: Path) -> tuple[list[dict], list[dict]]:
        """Split calls_data.json entries into (eligible, skipped) before any
        transcript content is read — honors is_private and missing-transcript
        calls up front."""
        calls = _load_small_json(root / "calls_data.json").get("calls", [])
        eligible, skipped = [], []
        for call in calls:
            if call.get("is_private"):
                skipped.append({**call, "skip_reason": "is_private"})
            elif not call.get("transcript"):
                skipped.append({**call, "skip_reason": "no_transcript"})
            else:
                eligible.append(call)
        return eligible, skipped

    def conversations(self, path: Path, account: str) -> Iterator[Conversation]:
        eligible, _skipped = self._eligible_calls(path)
        for call in eligible:
            transcript_path = path / call["transcript"]["filename"]
            raw = _load_small_json(transcript_path)
            yield self._conversation_from_call(raw, call, account, transcript_path)

    def _merge_segments(self, segments: list[dict]) -> list[dict]:
        turns: list[dict] = []
        for seg in segments:
            role = seg["role"]
            if role not in ("user", "assistant"):
                raise ValueError(f"unknown role {role!r} in sesame segment")
            if turns and turns[-1]["role"] == role:
                turn = turns[-1]
                turn["texts"].append(seg["content"])
                turn["end_time_ms"] = seg["end_time_ms"]
                turn["interrupted"] = turn["interrupted"] or bool(seg.get("interrupted"))
                turn["merged_count"] += 1
            else:
                turns.append(
                    {
                        "role": role,
                        "texts": [seg["content"]],
                        "start_time_ms": seg["start_time_ms"],
                        "end_time_ms": seg["end_time_ms"],
                        "interrupted": bool(seg.get("interrupted", False)),
                        "merged_count": 1,
                    }
                )
        return turns

    def _conversation_from_call(
        self, raw: dict, call: dict, account: str, source_path: Path
    ) -> Conversation:
        call_number = call["call_number"]
        native_id = str(call_number)
        call_start = datetime.fromisoformat(call["created_at"])

        segments = raw.get("conversation") or []
        declared_total = raw.get("total_segments")
        actual_total = len(segments)

        turns = self._merge_segments(segments)
        merge_count = sum(t["merged_count"] - 1 for t in turns)

        messages: list[Message] = []
        for i, t in enumerate(turns):
            text = " ".join(p for p in t["texts"] if p)
            if t["interrupted"]:
                text = f"{text} [interrupted]"
            ts = _iso(call_start + timedelta(milliseconds=t["start_time_ms"]))
            messages.append(Message(i=i, role=t["role"], text=text, ts=ts, model=None))

        return Conversation(
            uid=f"sesame:{account}:{native_id}",
            platform="sesame",
            account=account,
            native_id=native_id,
            kind="voice_call",
            title=None,
            created_at=call.get("created_at"),
            updated_at=call.get("ended_at"),
            model=None,
            project_ref=None,
            messages=messages,
            meta={
                "asr_confidence_note": _ASR_CONFIDENCE_NOTE,
                "total_segments_declared": declared_total,
                "total_segments_actual": actual_total,
                "segment_count_mismatch": actual_total - (declared_total or 0),
                "segment_merge_count": merge_count,
                "original_trace_duration_ms": raw.get("original_trace_duration_ms"),
                "source_path": str(source_path),
            },
        )

    # ------------------------------------------------------------------- artifacts

    def artifacts(self, path: Path, account: str) -> Iterator[Artifact]:
        yield from self._profile_artifacts(path, account)
        yield from self._memory_artifacts(path, account)

    def _profile_artifacts(self, path: Path, account: str) -> Iterator[Artifact]:
        user_info_path = path / "user_info.json"
        if not user_info_path.exists():
            return
        raw = _load_small_json(user_info_path)
        yield Artifact(
            uid=f"sesame:{account}:profile:{account}",
            platform="sesame",
            account=account,
            kind="profile",
            source_path=str(user_info_path),
            title=raw.get("display_name"),
            created_at=None,
            text=orjson.dumps(raw).decode(),
            meta=dict(raw),
        )

    def _memory_artifacts(self, path: Path, account: str) -> Iterator[Artifact]:
        for f in sorted(path.glob("*_memory.txt")):
            text = f.read_text(encoding="utf-8")
            yield Artifact(
                uid=f"sesame:{account}:memory:{f.stem}",
                platform="sesame",
                account=account,
                kind="memory",
                source_path=str(f),
                title=f.stem,
                created_at=None,
                text=text,
                meta={},
            )
