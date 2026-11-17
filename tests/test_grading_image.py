from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import docker
from utils.docker_utils import build_container


class GradingImageTests(unittest.TestCase):
    def test_grading_build_selects_cpu_dockerfile_without_cluster_proxy(self):
        client = Mock()
        client.images.list.return_value = []
        client.images.build.return_value = (Mock(), [])
        client.containers.get.side_effect = docker.errors.NotFound("missing")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".dockerignore").write_text(".env\n*.csv\n")
            (root / "Dockerfile.imo_grading").write_text("FROM python:3.12-slim\n")
            container = build_container(client, directory, "hyperagents", "review",
                                        domains=["imo_grading"], verbose=False)
            self.assertIs(container, client.containers.run.return_value)
            build = client.images.build.call_args.kwargs
            self.assertEqual(build["dockerfile"], "Dockerfile.imo_grading")
            self.assertEqual(build["tag"], "hyperagents-imo-grading-v2")
            self.assertNotIn("buildargs", build)
            self.assertNotIn("network_mode", build)
            run = client.containers.run.call_args.kwargs
            self.assertEqual(run["image"], build["tag"])
            self.assertNotIn("device_requests", run)
            self.assertEqual(run["volumes"][str(root.resolve())]["bind"], "/hyperagents")


if __name__ == "__main__":
    unittest.main()
