"""Fixtures must be hand-authored placeholders: no real identifiers, no long text."""
import json
import unittest

from helpers import FIXTURES

FORBIDDEN = ("tjbryant", "roadpiratefilms", "gmail.com")
MAX_LEN = 300


def _json_strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield k
            yield from _json_strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _json_strings(v)


class FixtureHygieneTest(unittest.TestCase):
    def test_fixtures_exist(self):
        self.assertTrue(any(p.is_file() for p in FIXTURES.rglob("*")))

    def test_no_real_identifiers_or_long_strings(self):
        for path in sorted(p for p in FIXTURES.rglob("*") if p.is_file()):
            with self.subTest(path=str(path.relative_to(FIXTURES))):
                self.assertFalse(any(s in str(path) for s in FORBIDDEN))
                raw = path.read_text(encoding="utf-8")
                for s in FORBIDDEN:
                    self.assertNotIn(s, raw.lower())
                if path.suffix == ".json":
                    strings = list(_json_strings(json.loads(raw)))
                else:
                    strings = raw.splitlines()
                for s in strings:
                    self.assertLessEqual(len(s), MAX_LEN, s[:60])


if __name__ == "__main__":
    unittest.main()
