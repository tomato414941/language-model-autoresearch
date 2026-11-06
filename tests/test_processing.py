import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import gpu_config
from tools import processing


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "local"
        self.remote = Path(self.temporary.name) / "remote copy"
        self.root.mkdir()
        self.remote.mkdir()
        self.config = self.root / "gpus.json"
        self.config.write_text(json.dumps(dict(active_gpu="rtx5090", gpus={
            "h100": {"gpu_name": "H100 SXM"}, "rtx5090": {"gpu_name": "RTX 5090"}})))
        for name, content in {".gitignore": ".env\n", ".env": "AUTORESEARCH_GPU=rtx5090\n",
                              "shared.py": "shared source\n", "helper with spaces.py": "helper\n",
                              "gpus/h100/train.py": "local h100\n",
                              "gpus/rtx5090/train.py": "local rtx5090\n"}.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        self.real_popen = subprocess.Popen
        self.addCleanup(patch.stopall)
        patch.object(processing, "ROOT", self.root).start()
        patch.object(gpu_config, "CONFIG_PATH", self.config).start()
        patch.dict(os.environ, {"PROCESSING_HOST": "test-host", "PROCESSING_DIR": str(self.remote),
                                "AUTORESEARCH_GPU": ""}).start()

    def run_main(self, *args):
        with patch.object(sys, "argv", ["processing.py", *args]), contextlib.redirect_stdout(io.StringIO()):
            processing.main()

    def test_deploys_selected_gpu_and_shared_sources_preserving_the_other_track(self):
        def popen(command, **kwargs):
            if command[0] == "ssh":
                return self.real_popen(["tar", "-xf", "-", "-C", str(self.remote)], **kwargs)
            return self.real_popen(command, **kwargs)

        with patch.object(processing.subprocess, "Popen", side_effect=popen):
            for selected, other in (("rtx5090", "h100"), ("h100", "rtx5090")):
                with self.subTest(gpu=selected):
                    protected = self.remote / "gpus" / other / "train.py"
                    protected.parent.mkdir(parents=True, exist_ok=True)
                    protected.write_text("independently maintained model\n")
                    self.run_main("--gpu", selected, "deploy")
                    self.assertEqual((self.remote / "gpus" / selected / "train.py").read_text(),
                                     f"local {selected}\n")
                    self.assertEqual(protected.read_text(), "independently maintained model\n")
                    for name in ("shared.py", "helper with spaces.py", "gpus.json", ".env"):
                        self.assertEqual((self.remote / name).read_bytes(), (self.root / name).read_bytes())

    def test_deploys_the_configured_default_gpu(self):
        expected = {b".gitignore", b"gpus.json", b"shared.py", b"helper with spaces.py",
                    b"gpus/rtx5090/train.py", b".env"}
        self.assertEqual(set(processing.deployment_files(gpu_config.active_gpu().identifier)), expected)

    def test_runs_with_the_selected_gpu_after_loading_the_remote_environment(self):
        (self.remote / ".env").write_text("AUTORESEARCH_GPU=rtx5090\n")
        observations = []

        def call(command):
            result = subprocess.run(["bash", "-c", command[-1]], capture_output=True, text=True, check=True)
            observations.append(result.stdout.splitlines())
            return result.returncode

        argument = "literal spaces;$(printf unexpected)"
        with patch.object(processing.subprocess, "call", side_effect=call):
            with self.assertRaises(SystemExit) as ended:
                self.run_main("--gpu", "h100", "run", sys.executable, "-c",
                              "import os,sys; print(os.environ['AUTORESEARCH_GPU']); print(sys.argv[1])", argument)
        self.assertEqual(ended.exception.code, 0)
        self.assertEqual(observations, [["h100", argument]])

    def test_propagates_the_process_gpu_selection_without_changing_the_default(self):
        (self.remote / ".env").write_text("AUTORESEARCH_GPU=rtx5090\n")
        observations = []

        def call(command):
            result = subprocess.run(["bash", "-c", command[-1]], capture_output=True, text=True, check=True)
            observations.append(result.stdout.strip())
            return result.returncode

        original = self.config.read_bytes()
        with patch.dict(os.environ, {"AUTORESEARCH_GPU": "h100"}), \
                patch.object(processing.subprocess, "call", side_effect=call):
            with self.assertRaises(SystemExit) as ended:
                self.run_main("run", sys.executable, "-c", "import os; print(os.environ['AUTORESEARCH_GPU'])")
        self.assertEqual(ended.exception.code, 0)
        self.assertEqual(observations, ["h100"])
        self.assertEqual(self.config.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
