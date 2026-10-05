"""The documentation check, wired into the suite.

Wrapped rather than duplicated: this runs the same script the shell would, so
there is one implementation and it cannot pass in one place and fail in
another.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class DocumentationTests(unittest.TestCase):
    def test_documentation_matches_the_code(self):
        """README, AGENTS.md and ROADMAP must not have drifted."""
        completed = subprocess.run(
            [sys.executable, "scripts/check_docs.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            "documentation has drifted:\n" + completed.stdout + completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
