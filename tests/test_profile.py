import unittest
from unittest import mock

import orjson

from src.profile import extract as ex
from src.profile import reduce as rd


def _body(obj, finish="stop"):
    return {"model": "m", "choices": [{"finish_reason": finish,
                                       "message": {"content": orjson.dumps(obj).decode()}}]}


def _batch():
    return {"account": "tjbryant", "conversations": [
        {"conv_uid": "c:1", "platform": "claude", "voice": False, "asr": False,
         "turns": [{"i": 0, "ts": "2026-01-01T00:00:00Z", "text": "I prefer tabs"},
                   {"i": 2, "ts": "2026-01-02T00:00:00Z", "text": "no, actually spaces"}]},
        {"conv_uid": "s:1", "platform": "sesame", "voice": True, "asr": True,
         "turns": [{"i": 1, "ts": "2026-02-01T00:00:00Z", "text": "hi"}]},
    ]}


class ExtractTest(unittest.TestCase):
    def test_payload_carries_content_only_as_json_data(self):
        p = ex.payload(_batch())
        self.assertNotIn("I prefer tabs", p["messages"][0]["content"])
        self.assertEqual(orjson.loads(p["messages"][1]["content"])["conversations"][0]["turns"][0]["text"],
                         "I prefer tabs")

    def test_invalid_claims_counted_and_dropped(self):
        good = {"claim": "Prefers spaces.", "category": "preference", "confidence": 0.8,
                "conv_uid": "c:1", "msg_i": [2, 0]}
        claims = [good,
                  {**good, "category": "vibes"},
                  {**good, "conv_uid": "nope"},
                  {**good, "msg_i": [1]},          # turn 1 belongs to another conversation
                  {**good, "confidence": 2},
                  {**good, "msg_i": []},
                  {**good, "conv_uid": "s:1", "msg_i": [1], "confidence": 1.0},
                  "not an object"]
        with mock.patch.object(ex, "chat", return_value=_body({"claims": claims})):
            lines, counts = ex.extract([_batch()])
        self.assertEqual(counts["claims"], 2)
        for reason in ("bad_category", "unknown_conv_uid", "msg_i_not_in_batch",
                       "bad_confidence", "bad_msg_i", "not_object"):
            self.assertEqual(counts[f"dropped:{reason}"], 1, reason)
        self.assertEqual(lines[0]["msg_i"], [0, 2])
        self.assertEqual(lines[0]["ts"], "2026-01-02T00:00:00Z")
        self.assertEqual(lines[1]["confidence"], round(1.0 * 0.8 * 0.7, 3))

    def test_unparseable_response_dropped_not_repaired(self):
        body = {"model": "m", "choices": [{"message": {"content": '{"claims": [ '}}]}
        with mock.patch.object(ex, "chat", return_value=body):
            lines, counts = ex.extract([_batch()])
        self.assertEqual(lines, [])
        self.assertEqual(counts["batch_invalid_response"], 1)
        self.assertEqual(counts["batch_retried"], 1)

    def test_truncated_batch_retried_and_exact_duplicates_dropped(self):
        good = {"claim": "Prefers spaces.", "category": "preference", "confidence": 0.8,
                "conv_uid": "c:1", "msg_i": [2]}
        bodies = [_body({"claims": [good] * 3}, finish="length"), _body({"claims": [good, good]})]
        with mock.patch.object(ex, "chat", side_effect=bodies) as chat:
            lines, counts = ex.extract([_batch()])
        self.assertEqual(chat.call_args_list[1].args[0]["temperature"], ex.RETRY_TEMPERATURE)
        self.assertEqual(len(lines), 1)
        self.assertEqual(counts["dropped:exact_duplicate"], 1)

    def test_batches_split_by_turn_and_account(self):
        convs = [
            {"uid": "x:tjbryant:a", "account": "tjbryant", "platform": "claude", "meta": {},
             "messages": [{"i": i, "role": r, "ts": None, "text": "x" * 50_000}
                          for i, r in enumerate(["user", "assistant", "user", "user"])]},
            {"uid": "x:roadpiratefilms:b", "account": "roadpiratefilms", "platform": "grok", "meta": {},
             "messages": [{"i": 0, "role": "user", "ts": None, "text": "hi"}]},
        ]
        with mock.patch.object(ex, "_conversations", return_value=iter(convs)):
            batches = ex.build_batches(["x:tjbryant:a", "x:roadpiratefilms:b"])
        self.assertEqual([b["account"] for b in batches], ["tjbryant", "tjbryant", "roadpiratefilms"])
        self.assertEqual([[t["i"] for c in b["conversations"] for t in c["turns"]] for b in batches],
                         [[0, 2], [3], [0]])


def _line(i, claim, ts, conf=0.5):
    return {"id": i, "claim": claim, "category": "preference", "confidence": conf,
            "account": "tjbryant", "conv_uid": f"c:{i}", "msg_i": [0], "ts": ts,
            "source": "extract", "model": "m"}


class ReduceTest(unittest.TestCase):
    def test_merge_and_supersede_by_recency(self):
        ledger = [_line("a", "Prefers tabs.", "2024-01-01T00:00:00Z", 0.6),
                  _line("b", "Likes tabs.", "2024-06-01T00:00:00Z", 0.5),
                  _line("c", "Prefers spaces.", "2026-01-01T00:00:00Z"),
                  _line("d", "Uses vim.", None),
                  {**_line("s", "memory text", None), "source": "seed"}]
        resp = {"duplicates": [["a", "b"], ["a", "zzz"]], "contradictions": [["c", "a"], ["d", "c"]]}
        with mock.patch.object(rd, "chat", return_value=_body(resp)):
            out, counts = rd.reduce(ledger)
        by = {r["id"]: r for r in out}
        self.assertEqual(set(by), {"b", "c", "d"})        # seed excluded; a merged into b
        self.assertEqual(by["b"]["claim"], "Likes tabs.")  # most recent member wins
        self.assertEqual(by["b"]["confidence"], 0.65)
        self.assertEqual(by["b"]["first_seen"], "2024-01-01T00:00:00Z")
        self.assertEqual(by["b"]["superseded_by"], "c")
        self.assertIsNone(by["c"]["superseded_by"])
        self.assertEqual(counts["dropped:bad_duplicate_group"], 1)
        self.assertEqual(counts["contradiction_unresolved"], 1)
        self.assertEqual(counts["live_claims"], 2)


class RedactTest(unittest.TestCase):
    def _reduced(self, claim, category="belief"):
        return {"id": claim, "claim": claim, "category": category, "confidence": 0.9,
                "account": "tjbryant", "ts": None, "sources": [{}], "superseded_by": None}

    def test_category_and_topic_filtered_from_render_inputs(self):
        from src.profile import render as rn
        reduced = [self._reduced("Believes he is being tracked inside his apartment."),
                   self._reduced("Researched IPS.", "surveillance_concern"),
                   self._reduced("Uses git to track dotfiles.", "technical"),
                   self._reduced("Prefers terse answers.", "preference")]
        claims = rn.render_inputs(reduced, [], ("tjbryant",), None)["claims"]
        self.assertCountEqual([c["claim"] for c in claims], ["Uses git to track dotfiles.", "Prefers terse answers."])

    def test_rendered_output_with_topic_is_refused(self):
        from src.profile import render as rn
        with mock.patch.object(rn, "PROFILE_DIR") as d:
            with self.assertRaises(rn.OpenRouterError):
                rn._write("core.md", "The user worries about being watched at home.")
            d.__truediv__.assert_not_called()

    def test_pii_claims_dropped_and_seed_lines_scrubbed(self):
        from src.profile import render as rn
        reduced = [self._reduced("Uses a@b.com for work.", "identity"),
                   self._reduced("Was born in March.", "identity"),
                   self._reduced("Earns little.", "finance"),
                   self._reduced("Prefers terse answers.", "preference")]
        seed = {"account": "tjbryant", "category": "artifact:memory", "title": "mem",
                "confidence": 0.9, "claim": "likes vim\nemail: x@y.org\n42 years old"}
        data = rn.render_inputs(reduced, [seed], ("tjbryant",), None)
        self.assertEqual([c["claim"] for c in data["claims"]], ["Prefers terse answers."])
        self.assertEqual(data["artifacts"][0]["text"], "likes vim\n[redacted]\n[redacted]")


if __name__ == "__main__":
    unittest.main()
