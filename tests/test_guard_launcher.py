import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "slime" / "scripts"


class GuardLauncherTests(unittest.TestCase):
    def test_importing_utils_does_not_load_training_arguments(self):
        source = SCRIPTS.parent
        result = subprocess.run(
            [
                sys.executable, "-c",
                "import sys, types; "
                "sys.modules['slime.utils.arguments'] = None; "
                "import slime.utils; "
                "args = types.ModuleType('slime.utils.arguments'); "
                "args.parse_args = object(); "
                "sys.modules[args.__name__] = args; "
                "from slime.utils import parse_args; "
                "assert parse_args is args.parse_args",
            ],
            cwd=source, text=True, capture_output=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.slime = self.root / "slime"
        self.scripts = self.slime / "scripts"
        self.scripts.mkdir(parents=True)
        self.events = self.root / "events"
        self.env = {
            **os.environ,
            "PATH": f"{self.root}:{os.environ['PATH']}",
            "EVENTS": str(self.events),
            "SAVE_DIR": str(self.root / "save"),
            "HF_HOME": str(self.root / "huggingface"),
            "GUARD_VENV": str(self.root / "guard-venv"),
        }

    def script(self, path, body):
        path.write_text("#!/usr/bin/env bash\n" + body)
        path.chmod(0o755)

    def run_script(self, command, **env):
        return subprocess.run(
            ["bash", "-c", command],
            env={**self.env, **env},
            text=True,
            capture_output=True,
            timeout=15,
        )

    def test_guard_dependencies_are_installed_only_in_guard_venv(self):
        shutil.copyfile(SCRIPTS / "runpod_util.sh", self.scripts / "runpod_util.sh")
        base = self.root / "base-python"
        self.script(
            base,
            'echo "base:$*" >> "$EVENTS"\n'
            'if [[ "$1" == "-m" && "$2" == "venv" ]]; then\n'
            '  mkdir -p "$GUARD_VENV/bin"\n'
            '  cp "$GUARD_STUB" "$GUARD_VENV/bin/python3"\n'
            'elif [[ "$*" == *"sys.base_prefix"* ]]; then echo /base\n'
            'elif [[ "$*" == *"sys.version_info"* ]]; then echo 3.12\n'
            'elif [[ "$*" == *"torch.__file__"* ]]; then echo /base/torch\n'
            'fi\n',
        )
        guard = self.root / "guard-stub"
        self.script(
            guard,
            'echo "guard:$*" >> "$EVENTS"\n'
            'if [[ "$1" == "-" && "$#" == 1 ]]; then exit 1; fi\n',
        )
        command = (
            f'set -e; source "{self.scripts}/runpod_util.sh"; '
            f'RUNPOD_PYTHON="{base}"; SLIME_ROOT="{self.slime}"; '
            'runpod_ensure_guard_environment; echo "$GUARD_PYTHON"'
        )
        result = self.run_script(command, GUARD_STUB=str(guard))
        self.assertEqual(result.returncode, 0, result.stderr)
        events = self.events.read_text().splitlines()
        self.assertTrue(any("base:-m venv --system-site-packages" in line for line in events))
        self.assertTrue(any("guard:-m pip --isolated install" in line for line in events))
        self.assertFalse(any("base:-m pip" in line for line in events))
        self.assertIn(str(self.root / "guard-venv" / "bin" / "python3"), result.stdout)

    def test_launcher_uses_guard_python_and_reports_early_exit(self):
        shutil.copyfile(SCRIPTS / "start_guard_service.sh", self.scripts / "start_guard_service.sh")
        base = self.root / "base-python"
        guard = self.root / "guard-python"
        self.script(base, 'echo "gpu-probe" >> "$EVENTS"\necho 1\n')
        self.script(guard, 'echo "guard:${CUDA_VISIBLE_DEVICES-unset}:$*" >> "$EVENTS"\nexit 3\n')
        self.script(self.root / "curl", "exit 1\n")
        (self.scripts / "runpod_util.sh").write_text(
            "runpod_ensure_environment() { RUNPOD_PYTHON=\"$BASE_PYTHON\"; }\n"
            "runpod_set_model_paths() { SAFETY_STREAM_MODEL_PATH=stream; "
            "SAFETY_GUARD_MODEL_PATH=gen; LLAMA_GUARD_MODEL_PATH=llama; }\n"
            "runpod_ensure_hf_models() { :; }\n"
            "runpod_ensure_guard_environment() { GUARD_PYTHON=\"$GUARD_STUB\"; }\n"
        )
        for device, expected_cuda in (("gpu", "0"), ("cpu", "")):
            with self.subTest(device=device):
                self.events.unlink(missing_ok=True)
                result = self.run_script(
                    f'bash "{self.scripts}/start_guard_service.sh"',
                    BASE_PYTHON=str(base),
                    GUARD_STUB=str(guard),
                    GUARD_DEVICE=device,
                    CUDA_VISIBLE_DEVICES="0",
                    GUARD_GPU_ID="0",
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Guard service exited before becoming ready", result.stdout + result.stderr)
                events = self.events.read_text()
                self.assertIn(f"guard:{expected_cuda}:-m slime.services.guard_service", events)
                self.assertEqual("gpu-probe" in events, device == "gpu")

    def test_guard_venv_cannot_be_the_training_environment(self):
        shutil.copyfile(SCRIPTS / "runpod_util.sh", self.scripts / "runpod_util.sh")
        venv = self.root / "shared-venv"
        subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
            check=True,
        )
        (self.root / "torch.py").write_text("")
        python = venv / "bin" / "python3"
        command = (
            f'set -e; source "{self.scripts}/runpod_util.sh"; '
            f'RUNPOD_PYTHON="{python}"; SLIME_ROOT="{self.slime}"; '
            'runpod_ensure_guard_environment'
        )
        result = self.run_script(command, GUARD_VENV=str(venv), PYTHONPATH=str(self.root))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Guard venv is incompatible", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
