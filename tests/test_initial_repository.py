import contextlib
from pathlib import Path
import subprocess
import tempfile
import unittest

from utils.constants import REPO_NAME
from utils.gl_utils import setup_initial_gen


class InitialRepositoryTests(unittest.TestCase):
    def git(self, root, *args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, encoding="utf-8"
        ).strip()

    def test_copy_commits_and_resets_without_touching_parent_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            self.git(project, "init", "--initial-branch=main")
            self.git(project, "config", "core.autocrlf", "false")
            self.git(project, "config", "user.name", "Review")
            self.git(project, "config", "user.email", "review@example.invalid")
            (project / ".gitignore").write_text("outputs/\n", encoding="utf-8")
            (project / "task_agent.py").write_text("value = 1\n", encoding="utf-8")
            (project / "domains/imo").mkdir(parents=True)
            (project / "domains/imo/grading_utils.py").write_text("# fixture\n")
            for name in ("Dockerfile.imo_grading", "requirements_imo_grading.txt", ".dockerignore"):
                (project / name).write_text("# fixture\n")
            self.git(project, "add", "--all")
            self.git(project, "commit", "-m", "parent baseline")
            parent_commit = self.git(project, "rev-parse", "HEAD")
            (project / "task_agent.py").write_text("value = 2\n", encoding="utf-8")
            self.git(project, "add", "task_agent.py")
            (project / "task_agent.py").write_text("value = 3\n", encoding="utf-8")
            parent_status = self.git(project, "status", "--porcelain")
            parent_index = self.git(project, "write-tree")

            output = project / "outputs/review"
            with contextlib.chdir(project):
                folder, commit = setup_initial_gen(str(output), ["imo_grading"], copy_eval=False)
            candidate = Path(folder)
            self.assertTrue((candidate / ".git").is_dir())
            self.assertTrue(Path(self.git(candidate, "rev-parse", "--show-toplevel")).samefile(candidate))
            self.assertEqual(len(commit), 40)
            self.assertEqual(self.git(candidate, "show", f"{commit}:task_agent.py"), "value = 3")
            self.assertEqual(self.git(candidate, "status", "--porcelain"), "")
            for name in ("Dockerfile.imo_grading", "requirements_imo_grading.txt", ".dockerignore"):
                self.assertTrue((candidate / name).is_file())
            (candidate / "task_agent.py").write_text("value = 4\n", encoding="utf-8")
            self.assertIn("+value = 4", self.git(candidate, "diff", commit))
            self.git(candidate, "reset", "--hard", commit)
            self.assertEqual((candidate / "task_agent.py").read_text(), "value = 3\n")
            self.assertEqual(self.git(project, "rev-parse", "HEAD"), parent_commit)
            self.assertEqual(self.git(project, "write-tree"), parent_index)
            self.assertEqual(self.git(project, "status", "--porcelain"), parent_status)
            self.assertEqual((project / "task_agent.py").read_text(), "value = 3\n")
            self.assertEqual(setup_initial_gen(str(output), ["imo_grading"], resume=True), (folder, commit))

    def test_resume_rejects_a_copy_that_would_use_the_parent_git_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "gen_initial" / REPO_NAME).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "independent Git repository"):
                setup_initial_gen(str(output), ["imo_grading"], resume=True)


if __name__ == "__main__":
    unittest.main()
