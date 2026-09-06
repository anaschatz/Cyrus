from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == "posix" and shutil.which("git") and shutil.which("sh"), "POSIX Git host required")
class PonytailHookTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.captured = self.root / "review-input"
        # Isolate fixture commits from user configuration, signing and hooks.
        self.env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        self.env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
                        GIT_AUTHOR_NAME="Hook Test", GIT_AUTHOR_EMAIL="hook@example.invalid",
                        GIT_COMMITTER_NAME="Hook Test", GIT_COMMITTER_EMAIL="hook@example.invalid")
        self.git("init", "-q")
        self.git("config", "core.hooksPath", str(self.root / "unused-hooks"))
        hooks = self.root / ".githooks"
        hooks.mkdir()
        for name in ("pre-commit", "pre-push"):
            shutil.copyfile(ROOT / ".githooks" / name, hooks / name)
        helper = hooks / "run-ponytail-review"
        helper.write_text('#!/bin/sh\ncat > "$PWD/review-input"\n')
        helper.chmod(0o700)

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.root, env=self.env, text=True,
                              capture_output=True, check=True, timeout=10).stdout.strip()

    def commit_file(self, name, text):
        (self.root / name).write_text(text)
        self.git("add", "--", name)
        self.git("commit", "-q", "-m", "Temporary hook fixture")
        return self.git("rev-parse", "HEAD")

    def hook(self, name, stdin="", env=None):
        return subprocess.run(["sh", str(self.root / ".githooks" / name), "origin"],
                              cwd=self.root, env=env or self.env, input=stdin, text=True,
                              capture_output=True, check=False, timeout=10)

    def test_pre_commit_reviews_only_staged_changes_and_propagates_review_failure(self):
        self.commit_file("item.txt", "base\n")
        (self.root / "item.txt").write_text("staged-value\n")
        self.git("add", "item.txt")
        (self.root / "item.txt").write_text("unstaged-value\n")
        result = self.hook("pre-commit")
        self.assertEqual(result.returncode, 0, result.stderr)
        review = self.captured.read_text()
        self.assertIn("+staged-value", review)
        self.assertNotIn("unstaged-value", review)
        (self.root / ".githooks/run-ponytail-review").write_text("#!/bin/sh\nexit 7\n")
        self.assertEqual(self.hook("pre-commit").returncode, 7)

    def test_pre_commit_does_not_review_a_failed_or_partial_diff(self):
        binary = self.root / "bin"
        binary.mkdir()
        git = binary / "git"
        git.write_text('#!/bin/sh\ncase "$1" in\n  rev-parse) printf \'%s\\n\' "$PWD" ;;\n  diff) printf \'%s\\n\' \'partial diff\'; exit 128 ;;\n  *) exit 128 ;;\nesac\n')
        git.chmod(0o700)
        result = self.hook("pre-commit", env={**self.env, "PATH": str(binary) + os.pathsep + self.env["PATH"]})
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.captured.exists(), "A failed diff must not reach the reviewer")

    def test_initial_push_reviews_all_files_in_the_outgoing_history(self):
        self.commit_file("first.txt", "first-commit-content\n")
        head = self.commit_file("second.txt", "second-commit-content\n")
        result = self.hook("pre-push", f"refs/heads/main {head} refs/heads/main {'0' * 40}\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        review = self.captured.read_text()
        self.assertIn("+first-commit-content", review)
        self.assertIn("+second-commit-content", review)

    def test_new_branch_excludes_existing_remote_history(self):
        base = self.commit_file("existing.txt", "already-published\n")
        self.git("update-ref", "refs/remotes/origin/main", base)
        head = self.commit_file("new.txt", "new-branch-content\n")
        result = self.hook("pre-push", f"refs/heads/feature {head} refs/heads/feature {'0' * 40}\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("+new-branch-content", self.captured.read_text())
        self.assertNotIn("already-published", self.captured.read_text())
        self.captured.unlink()
        self.git("update-ref", "refs/remotes/origin/main", head)
        result = self.hook("pre-push", f"refs/heads/alias {head} refs/heads/alias {'0' * 40}\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.captured.exists())

    def test_existing_branch_reviews_outgoing_delta_and_deletion_needs_no_diff(self):
        base = self.commit_file("item.txt", "base\n")
        head = self.commit_file("item.txt", "changed\n")
        result = self.hook("pre-push", f"refs/heads/main {head} refs/heads/main {base}\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("+changed", self.captured.read_text())
        self.captured.unlink()
        result = self.hook("pre-push", f"(delete) {'0' * 40} refs/heads/main {head}\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.captured.exists())


if __name__ == "__main__":
    unittest.main()
