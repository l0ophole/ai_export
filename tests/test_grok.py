import shutil
import tempfile
import unittest
from pathlib import Path

import orjson

from helpers import FIXTURES, assert_messages_utc_z, assert_utc_z, roles, run, texts
from src.adapters.grok import GrokAdapter


class GrokFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.accounts, cls.convs, cls.arts = run(GrokAdapter())
        cls.by_id = {c["native_id"]: c for c in cls.convs}

    def test_counts(self):
        self.assertEqual(self.accounts, ["acct_a"])
        self.assertEqual(len(self.convs), 2)
        self.assertEqual(sum(len(c["messages"]) for c in self.convs), 6)

    def test_leaf_response_id_honored(self):
        c = self.by_id["g-conv-1"]
        self.assertFalse(c["meta"]["leaf_response_id_was_null"])
        self.assertEqual(roles(c), ["user", "assistant", "user", "assistant"])
        self.assertEqual(texts(c), ["u1", "a1", "u2", "a2-chosen"])
        self.assertEqual(c["meta"]["unreachable_node_count"], 1)
        self.assertEqual(c["meta"]["dropped_content_blocks"], {"file_attachments": 1})

    def test_null_leaf_uses_latest_childless(self):
        c = self.by_id["g-conv-2"]
        self.assertTrue(c["meta"]["leaf_response_id_was_null"])
        self.assertEqual(roles(c), ["user", "assistant"])
        self.assertEqual(texts(c), ["u1", "a1-new"])
        self.assertEqual(c["meta"]["unreachable_node_count"], 1)

    def test_mongo_date_to_utc_z(self):
        self.assertEqual(self.by_id["g-conv-1"]["messages"][0]["ts"], "2026-09-01T10:00:00Z")
        for c in self.convs:
            assert_utc_z(self, c["created_at"])
            assert_utc_z(self, c["updated_at"])
            assert_messages_utc_z(self, c)

    def test_artifacts(self):
        self.assertEqual([a["kind"] for a in self.arts], ["project_instructions", "profile"])
        media = self.arts[1]
        self.assertEqual(media["text"], "")
        self.assertEqual([p["id"] for p in media["meta"]["posts"]], ["mp1", "mp2"])

    def test_nonempty_tasks_raises(self):
        src = FIXTURES / "grok" / "acct_a" / "exported/ttl/x/export_data/y/prod-grok-backend.json"
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "prod-grok-backend.json"
            shutil.copy(src, dst)
            raw = orjson.loads(dst.read_bytes())
            raw["tasks"] = [{"id": "t1"}]
            dst.write_bytes(orjson.dumps(raw))
            with self.assertRaises(NotImplementedError):
                list(GrokAdapter().artifacts(dst, "acct_a"))


if __name__ == "__main__":
    unittest.main()
