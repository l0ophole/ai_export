import unittest

from helpers import assert_messages_utc_z, assert_utc_z, roles, run, texts
from src.adapters.copilot import CopilotAdapter


class CopilotFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.accounts, cls.convs, cls.arts = run(CopilotAdapter())

    def test_counts(self):
        self.assertEqual(self.accounts, ["acct_a"])
        self.assertEqual(len(self.convs), 3)
        self.assertEqual(sum(len(c["messages"]) for c in self.convs), 8)

    def test_title_collision_split_by_gap(self):
        same = [c for c in self.convs if c["title"] == "Same Title"]
        self.assertEqual([c["meta"]["title_split_index"] for c in same], [0, 1])
        self.assertEqual(texts(same[1]), ["u1-later", "a1-later"])
        self.assertNotEqual(same[0]["native_id"], same[1]["native_id"])

    def test_multiline_quoted_message(self):
        first = self.convs[0]
        self.assertEqual(roles(first), ["user", "assistant", "user", "assistant"])
        self.assertEqual(
            texts(first),
            ["u1", 'a1 line one\na1 line two, with comma\n\na1 "quoted" para', "u2", "a2"],
        )

    def test_human_before_ai_on_tie(self):
        other = [c for c in self.convs if c["title"] == "Other Title"][0]
        self.assertEqual(roles(other), ["user", "assistant"])
        self.assertEqual(texts(other), ["u1-tie", "a1-tie"])

    def test_chicago_to_utc_z(self):
        self.assertEqual(self.convs[0]["created_at"], "2026-01-15T16:00:00Z")  # CST
        other = [c for c in self.convs if c["title"] == "Other Title"][0]
        self.assertEqual(other["created_at"], "2026-07-04T14:00:00Z")  # CDT
        for c in self.convs:
            assert_utc_z(self, c["created_at"])
            assert_utc_z(self, c["updated_at"])
            assert_messages_utc_z(self, c)

    def test_no_artifacts(self):
        self.assertEqual(self.arts, [])


if __name__ == "__main__":
    unittest.main()
