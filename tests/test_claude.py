import unittest

from helpers import assert_messages_utc_z, assert_utc_z, roles, run, texts
from src.adapters.claude import ClaudeAdapter


class ClaudeFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.accounts, cls.convs, cls.arts = run(ClaudeAdapter(exclusions={}))
        cls.by_id = {c["native_id"]: c for c in cls.convs}

    def test_discover_accounts(self):
        self.assertEqual(self.accounts, ["acct_a", "acct_b"])

    def test_counts(self):
        self.assertEqual(len(self.convs), 4)
        self.assertEqual(sum(len(c["messages"]) for c in self.convs), 10)
        self.assertEqual([c["kind"] for c in self.convs], ["chat", "chat", "design_chat", "chat"])

    def test_branch_linearized_from_latest_leaf(self):
        c = self.by_id["c1000000-0000-4000-8000-000000000001"]
        self.assertEqual(roles(c), ["user", "assistant", "user", "assistant"])
        self.assertEqual(texts(c), ["u1", "a1\n\na1b", "u2-new", "a2-new"])
        self.assertEqual(c["meta"]["unreachable_node_count"], 2)
        self.assertEqual(c["meta"]["summary"], "s1")

    def test_dropped_blocks_counted(self):
        c = self.by_id["c1000000-0000-4000-8000-000000000001"]
        self.assertEqual(c["meta"]["dropped_content_blocks"], {"voice_note": 1, "tool_use": 1, "tool_result": 1})

    def test_voice_mode_flag(self):
        self.assertTrue(self.by_id["c1000000-0000-4000-8000-000000000001"]["meta"]["voice_mode"])
        self.assertFalse(self.by_id["c1000000-0000-4000-8000-000000000002"]["meta"]["voice_mode"])

    def test_exclusions_skip_chat_and_design_chat(self):
        excluded = {
            "claude:acct_a:c1000000-0000-4000-8000-000000000001": "test",
            "claude:acct_a:d1000000-0000-4000-8000-000000000001": "test",
        }
        _, convs, _ = run(ClaudeAdapter(exclusions=excluded))
        self.assertEqual(len(convs), 2)
        self.assertFalse({c["uid"] for c in convs} & set(excluded))

    def test_legacy_text_fallback(self):
        c = self.by_id["c1000000-0000-4000-8000-000000000002"]
        self.assertEqual(texts(c), ["u1", "a1"])
        self.assertEqual(c["meta"]["dropped_content_blocks"], {})

    def test_design_chat(self):
        c = self.by_id["d1000000-0000-4000-8000-000000000001"]
        self.assertEqual(c["kind"], "design_chat")
        self.assertEqual(c["project_ref"], "claude:acct_a:p1000000-0000-4000-8000-000000000001")
        self.assertEqual(roles(c), ["user", "assistant"])
        self.assertEqual(texts(c), ["du1", "da1"])
        self.assertEqual(c["meta"]["dropped_content_blocks"], {"thinking": 1, "tool_call": 1})

    def test_timestamps_utc_z(self):
        for c in self.convs:
            assert_utc_z(self, c["created_at"])
            assert_utc_z(self, c["updated_at"])
            assert_messages_utc_z(self, c)

    def test_artifact_kinds(self):
        kinds = [a["kind"] for a in self.arts]
        self.assertEqual(
            kinds,
            ["project_instructions", "project_doc", "project_doc", "memory", "memory", "memory",
             "reflection", "reflection", "profile"],
        )
        self.assertEqual([a["text"] for a in self.arts if a["kind"] == "memory"], ["mem1", "pmem1", "mf1"])
        self.assertTrue(all(a["account"] == "acct_a" for a in self.arts))


if __name__ == "__main__":
    unittest.main()
