import unittest

from helpers import assert_messages_utc_z, assert_utc_z, roles, run, texts
from src.adapters.deepseek import DeepseekAdapter


class DeepseekFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.accounts, cls.convs, cls.arts = run(DeepseekAdapter())
        cls.by_id = {c["native_id"]: c for c in cls.convs}

    def test_counts(self):
        self.assertEqual(self.accounts, ["acct_a"])
        self.assertEqual(len(self.convs), 2)
        self.assertEqual(sum(len(c["messages"]) for c in self.convs), 6)

    def test_fork_linearized_from_latest_leaf(self):
        c = self.by_id["ds-conv-1"]
        self.assertEqual(roles(c), ["user", "assistant", "user", "assistant"])
        self.assertEqual(texts(c), ["u1", "a1", "u2", "a2-new"])
        self.assertEqual(c["meta"]["unreachable_node_count"], 1)

    def test_think_kept_out_of_text(self):
        c = self.by_id["ds-conv-1"]
        self.assertEqual(c["meta"]["think_by_index"], {"1": "th1"})
        self.assertEqual(c["meta"]["dropped_content_blocks"], {"SEARCH": 1})

    def test_empty_fragments_is_assistant(self):
        c = self.by_id["ds-conv-2"]
        self.assertEqual(roles(c), ["user", "assistant"])
        self.assertEqual(texts(c), ["u1", ""])

    def test_timestamps_converted_to_utc_z(self):
        c = self.by_id["ds-conv-1"]
        self.assertEqual(c["created_at"], "2025-09-17T12:00:00Z")
        self.assertEqual(c["messages"][-1]["ts"], "2025-09-17T12:03:00Z")
        for c in self.convs:
            assert_utc_z(self, c["created_at"])
            assert_utc_z(self, c["updated_at"])
            assert_messages_utc_z(self, c)

    def test_artifacts(self):
        self.assertEqual([a["kind"] for a in self.arts], ["profile"])


if __name__ == "__main__":
    unittest.main()
