import unittest

from helpers import FIXTURES, assert_messages_utc_z, assert_utc_z, roles, run, texts
from src.adapters.sesame import SesameAdapter


class SesameFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = SesameAdapter()
        cls.accounts, cls.convs, cls.arts = run(cls.adapter)
        cls.by_id = {c["native_id"]: c for c in cls.convs}

    def test_counts(self):
        self.assertEqual(self.accounts, ["acct_a"])
        self.assertEqual(sorted(self.by_id), ["1", "3"])
        self.assertEqual(sum(len(c["messages"]) for c in self.convs), 6)
        self.assertTrue(all(c["kind"] == "voice_call" for c in self.convs))

    def test_private_and_missing_transcript_skipped(self):
        # call_2's transcript file does not exist: reading it would raise.
        _eligible, skipped = self.adapter._eligible_calls(FIXTURES / "sesame")
        self.assertEqual(
            [(s["call_number"], s["skip_reason"]) for s in skipped],
            [(2, "is_private"), (4, "no_transcript")],
        )

    def test_segments_merged_and_interrupted_carried(self):
        c = self.by_id["1"]
        self.assertEqual(roles(c), ["user", "assistant", "user", "assistant"])
        self.assertEqual(texts(c), ["u1a u1b", "a1a a1b [interrupted]", "u2", "a2"])
        self.assertEqual(c["meta"]["segment_merge_count"], 2)
        self.assertEqual(c["meta"]["total_segments_actual"], 6)
        self.assertIn("asr_confidence_note", c["meta"])

    def test_assistant_first_call(self):
        self.assertEqual(roles(self.by_id["3"]), ["assistant", "user"])

    def test_timestamps(self):
        self.assertEqual(self.by_id["1"]["messages"][1]["ts"], "2026-07-07T15:35:03Z")
        self.assertEqual(self.by_id["1"]["created_at"], "2026-07-07T15:35:00Z")
        self.assertEqual(self.by_id["1"]["updated_at"], "2026-07-07T15:40:00Z")
        for c in self.convs:
            assert_utc_z(self, c["created_at"])
            assert_utc_z(self, c["updated_at"])
            assert_messages_utc_z(self, c)

    def test_artifacts(self):
        self.assertEqual([a["kind"] for a in self.arts], ["profile", "memory"])
        self.assertEqual(self.arts[1]["text"], "mem line 1\nmem line 2\n")


if __name__ == "__main__":
    unittest.main()
