import io
from pathlib import Path, PureWindowsPath
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from utils.docker_utils import copy_from_container, copy_to_container, create_archive


class DockerTransferTests(unittest.TestCase):
    def container(self):
        container = Mock()
        container.exec_run.return_value = SimpleNamespace(exit_code=0, output=b"")
        container.put_archive.return_value = True
        return container

    def archive(self, entries):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as archive:
            for name, content in entries.items():
                member = tarfile.TarInfo(name)
                if content is None:
                    member.type = tarfile.DIRTYPE
                    archive.addfile(member)
                else:
                    member.size = len(content)
                    archive.addfile(member, io.BytesIO(content))
        return stream.getvalue()

    def test_upload_uses_posix_paths_and_preserves_spaces_and_nested_files(self):
        container = self.container()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "host feedback"
            (source / "nested").mkdir(parents=True)
            (source / "nested/chat.md").write_bytes(b"feedback")
            copy_to_container(container, source, PureWindowsPath("/tmp/agent feedback"), verbose=False)
            container.exec_run.assert_called_once_with(["mkdir", "-p", "/tmp"])
            destination, payload = container.put_archive.call_args.args
            self.assertEqual(destination, "/tmp")
            with tarfile.open(fileobj=payload) as archive:
                self.assertEqual(archive.extractfile("agent feedback/nested/chat.md").read(), b"feedback")
                self.assertTrue(all("\\" not in m.name for m in archive.getmembers()))
                self.assertTrue(all(m.uid == m.gid == 0 for m in archive.getmembers()))
            container.reset_mock()
            copy_to_container(container, source / "nested/chat.md", "/tmp/agent feedback/renamed.md", verbose=False)
            container.exec_run.assert_called_once_with(["mkdir", "-p", "/tmp/agent feedback"])
            destination, payload = container.put_archive.call_args.args
            self.assertEqual(destination, "/tmp/agent feedback")
            with tarfile.open(fileobj=payload) as archive:
                self.assertEqual(archive.extractfile("renamed.md").read(), b"feedback")
            with tarfile.open(fileobj=io.BytesIO(create_archive(source))) as archive:
                self.assertTrue(all("\\" not in m.name for m in archive.getmembers()))

    def test_download_file_is_renamed_without_shell_stat(self):
        container = self.container()
        blob = self.archive({"meta_usage.jsonl": b'{"requests": 1}\n'})
        container.get_archive.return_value = (iter([blob]), {"name": "meta_usage.jsonl"})
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "generation 1/usage.jsonl"
            copy_from_container(container, PureWindowsPath("/tmp/meta_usage.jsonl"), destination, verbose=False)
            self.assertEqual(destination.read_bytes(), b'{"requests": 1}\n')
        container.get_archive.assert_called_once_with("/tmp/meta_usage.jsonl")
        container.exec_run.assert_not_called()

    def test_repeated_directory_exports_merge_existing_generation_records(self):
        container = self.container()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "agent_output"
            destination.mkdir()
            (destination / "existing.log").write_bytes(b"keep")
            for content in (b"first export", b"second export"):
                blob = self.archive({"agent_output": None,
                                     "agent_output/model_patch.diff": content,
                                     "agent_output/nested": None,
                                     "agent_output/nested/chat.md": b"chat"})
                container.get_archive.return_value = (iter([blob]), {"name": "agent_output"})
                copy_from_container(container, "/tmp/agent_output", destination, verbose=False)
                self.assertEqual((destination / "model_patch.diff").read_bytes(), content)
                self.assertEqual((destination / "nested/chat.md").read_bytes(), b"chat")
                self.assertEqual((destination / "existing.log").read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
