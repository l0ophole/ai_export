import shutil
import tempfile
import unittest
from pathlib import Path

from helpers import FIXTURES, assert_messages_utc_z, assert_utc_z, roles, run, texts
from src.adapters.gemini import GeminiAdapter


class GeminiFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.accounts, cls.convs, cls.arts = run(GeminiAdapter())
        cls.by_id = {c["native_id"]: c for c in cls.convs}

    def test_counts(self):
        self.assertEqual(self.accounts, ["acct_a"])
        self.assertEqual(sorted(self.by_id), ["aaa1", "ddd4"])
        self.assertEqual(sum(len(c["messages"]) for c in self.convs), 8)

    def test_details_ids_merged_by_union_find(self):
        c = self.by_id["aaa1"]
        self.assertEqual(c["meta"]["aliased_conversation_ids"], ["aaa1", "bbb2", "ccc3"])
        self.assertEqual(roles(c), ["user", "assistant"] * 3)
        t = texts(c)
        self.assertTrue(t[0].startswith("u1"))
        self.assertEqual(t[1:], ["a1", "u2", "a2", "u3", "a3"])
        self.assertEqual(c["meta"]["verb_counts"], {"Branched": 1, "Prompted": 2})
        self.assertEqual(c["meta"]["attachments"], ["img_a.png"])

    def test_timestamps_localized_not_literal_cdt(self):
        # "CDT" in January is a Takeout quirk; real offset is CST (-6).
        self.assertEqual(self.by_id["aaa1"]["created_at"], "2026-01-10T21:00:00Z")
        self.assertEqual(self.by_id["ddd4"]["created_at"], "2025-07-04T15:00:00Z")
        for c in self.convs:
            assert_utc_z(self, c["created_at"])
            assert_utc_z(self, c["updated_at"])
            assert_messages_utc_z(self, c)

    def test_artifacts(self):
        self.assertEqual([a["kind"] for a in self.arts], ["project_doc", "project_doc", "profile"])
        canvas, app, manifest = self.arts
        self.assertEqual((canvas["title"], canvas["text"]), ("Canvas One", "canvas body c1"))
        assert_utc_z(self, canvas["created_at"])
        self.assertEqual(app["title"], "Canvas App")
        self.assertEqual(manifest["meta"]["telemetry_entry_counts"], {"Used": 1, "Cleared": 1})
        self.assertEqual(manifest["meta"]["referenced_media_files"], ["img_a.png"])
        self.assertEqual(manifest["meta"]["orphaned_media_files"], ["orphan_b.txt"])

    def test_telemetry_not_emitted(self):
        all_text = [m["text"] for c in self.convs for m in c["messages"]]
        self.assertFalse(any("Gemini Apps" in t or "feedback" in t for t in all_text))

    def test_non_cdt_abbreviation_raises(self):
        src = FIXTURES / "gemini" / "acct_a" / "MyActivity.html"
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td) / "MyActivity.html"
            shutil.copy(src, dst)
            dst.write_text(dst.read_text(encoding="utf-8").replace("PM CDT", "PM CST", 1), encoding="utf-8")
            with self.assertRaises(NotImplementedError):
                list(GeminiAdapter().conversations(Path(td), "acct_a"))


if __name__ == "__main__":
    unittest.main()
