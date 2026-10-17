from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path

from prepare_public_archive import (
    ArchiveError,
    _destination_is_stageable,
    _redact_text,
    build_plan,
    collect_allowlisted_files,
)


class ArchiveHelperTests(unittest.TestCase):
    def _temporary_directory(self):
        # Python's tempfile uses restrictive Windows ACLs in this workspace.
        temporary = Path.cwd() / f"archive-helper-case-{uuid.uuid4().hex}"
        temporary.mkdir()
        self.addCleanup(shutil.rmtree, temporary, ignore_errors=True)
        return temporary

    def test_evidence_projection_redacts_paths_and_credentials(self) -> None:
        source = 'log=C:\\Users\\alice\\private\\run.json token="sensitive-value"\n'
        public, redactions = _redact_text(source)
        self.assertNotIn("C:\\Users\\alice", public)
        self.assertNotIn("sensitive-value", public)
        self.assertEqual(redactions["windows_absolute_path"], 1)
        self.assertEqual(redactions["credential_assignment"], 1)

    def test_allowlist_preserves_package_cli_paths_and_ignores_outputs(self) -> None:
        root = self._temporary_directory()
        (root / "package" / "civilian").mkdir(parents=True)
        (root / "package" / "tests").mkdir(parents=True)
        (root / "outputs").mkdir()
        (root / ".deps").mkdir()
        (root / "package" / "civilian" / "gain_runner.py").write_text("pass\n", encoding="utf-8")
        (root / "package" / "tests" / "test_gain_runner.py").write_text("pass\n", encoding="utf-8")
        (root / "outputs" / "result.json").write_text("{}\n", encoding="utf-8")
        (root / ".deps" / "private.whl").write_text("ignored\n", encoding="utf-8")

        selected = {item.relative_path for item in collect_allowlisted_files(root)}

        self.assertIn("package/civilian/gain_runner.py", selected)
        self.assertIn("package/tests/test_gain_runner.py", selected)
        self.assertNotIn("outputs/result.json", selected)
        self.assertNotIn(".deps/private.whl", selected)

    def test_apply_guard_allows_git_metadata_but_refuses_existing_files(self) -> None:
        destination = self._temporary_directory()
        (destination / ".git").mkdir()
        self.assertTrue(_destination_is_stageable(destination))
        (destination / "README.md").write_text("existing\n", encoding="utf-8")
        self.assertFalse(_destination_is_stageable(destination))

    def test_non_utf8_source_is_rejected_before_staging(self) -> None:
        root = self._temporary_directory()
        (root / "package" / "civilian").mkdir(parents=True)
        (root / "package" / "civilian" / "gain_runner.py").write_bytes(b"\xff")
        with self.assertRaises(ArchiveError):
            build_plan(root, root / "archive")


if __name__ == "__main__":
    unittest.main()
