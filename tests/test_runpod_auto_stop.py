import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "slime" / "scripts"


class RunpodAutoStopTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.scripts = self.root / "slime" / "scripts"
        self.scripts.mkdir(parents=True)
        self.events = self.root / "events"
        for name in ("runpod_tmux.sh", "runpod.sh", "upload_runpod.sh"):
            shutil.copyfile(SCRIPTS / name, self.scripts / name)
        self.write_script(
            "runpod_stop.sh",
            'echo "$1" >> "$EVENTS"\n'
            'if [[ "$1" == "--check" ]]; then exit "${CHECK_STATUS:-0}"; fi\n'
            'exit "${STOP_STATUS:-0}"\n',
        )
        self.write_script(
            "runpod_download.sh",
            'echo "download" >> "$EVENTS"\nexit "${WORK_STATUS:-0}"\n',
        )
        self.write_script("curl", "exit 1\n", directory=self.root)
        self.env = {
            **os.environ,
            "PATH": f"{self.root}:{os.environ['PATH']}",
            "EVENTS": str(self.events),
            "SAVE_DIR": str(self.root / "save"),
            "HF_TOKEN": "test",
        }
        self.env.pop("RUNPOD_AUTO_STOP", None)
        self.env.pop("RUNPOD_REUSE_RAY", None)

    def write_script(self, name, body, directory=None):
        path = (directory or self.scripts) / name
        path.write_text("#!/usr/bin/env bash\n" + body)
        path.chmod(0o755)

    def run_script(self, name, *args, **env):
        return subprocess.run(
            ["bash", str(self.scripts / name), *args],
            env={**self.env, **env},
            text=True,
            capture_output=True,
            timeout=10,
        )

    def event_list(self):
        return self.events.read_text().splitlines() if self.events.exists() else []

    def test_worker_stops_after_download_and_opted_in_preflight(self):
        self.write_script(
            "runpod.sh",
            'echo "run:$CONFIRM_RUN:$RUNPOD_START_GUARD" >> "$EVENTS"\n'
            'exit "${WORK_STATUS:-0}"\n',
        )
        for mode, expected in (
            ("download", ["download"]),
            ("preflight", ["run:NO:YES"]),
        ):
            for status in ("0", "7"):
                with self.subTest(mode=mode, status=status):
                    self.events.unlink(missing_ok=True)
                    result = self.run_script(
                        "runpod_tmux.sh", "--worker", mode,
                        "YES" if mode == "preflight" else "", "YES",
                        WORK_STATUS=status,
                    )
                    self.assertEqual(result.returncode, int(status), result.stderr)
                    self.assertEqual(self.event_list(), ["--check", *expected, "--stop"])

    def test_training_worker_stops_on_success_and_failure(self):
        self.write_script(
            "runpod.sh",
            'echo "run:$CONFIRM_RUN:$RUNPOD_START_GUARD:$RUNPOD_AUTO_STOP" >> "$EVENTS"\n'
            'exit "${WORK_STATUS:-0}"\n',
        )
        for status, message in (("0", "Training completed."), ("7", "train failed with exit code 7")):
            with self.subTest(status=status):
                self.events.unlink(missing_ok=True)
                result = self.run_script(
                    "runpod_tmux.sh", "--worker", "train", "YES", "YES", WORK_STATUS=status,
                )
                self.assertIn(message, result.stdout + result.stderr)
                self.assertEqual(result.returncode, int(status))
                self.assertEqual(self.event_list(), ["--check", "run:YES:YES:NO", "--stop"])

    def test_training_worker_can_disable_auto_stop(self):
        self.write_script("runpod.sh", 'echo "run:$RUNPOD_AUTO_STOP" >> "$EVENTS"\n')
        result = self.run_script("runpod_tmux.sh", "--worker", "train", "NO", "YES")
        self.assertIn("Training completed.", result.stdout)
        self.assertEqual(self.event_list(), ["run:NO"])

    def test_api_check_failure_still_requests_stop(self):
        result = self.run_script(
            "runpod_tmux.sh", "--worker", "preflight", "YES", "NO", CHECK_STATUS="9"
        )
        self.assertEqual(result.returncode, 9)
        self.assertEqual(self.event_list(), ["--check", "--stop"])

    def test_stop_failure_is_reported(self):
        result = self.run_script(
            "runpod_tmux.sh", "--worker", "download", "YES", "NO", STOP_STATUS="1"
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("Stop this pod manually", result.stdout + result.stderr)

    def test_training_requires_confirmation(self):
        result = self.run_script("runpod_tmux.sh", "session", "train", "YES", CONFIRM_RUN="NO")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CONFIRM_RUN=YES", result.stderr)
        result = self.run_script("runpod_tmux.sh", "--worker", "train", "YES", "NO")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.event_list(), [])

    def test_uploader_requires_and_forwards_training_confirmation(self):
        self.write_script("ssh", 'echo "ssh:$*" >> "$EVENTS"\n', directory=self.root)
        self.write_script("rsync", 'echo "rsync" >> "$EVENTS"\n', directory=self.root)
        env = {
            "RUNPOD_SSH_HOST": "pod",
            "RUNPOD_MODE": "train",
            "RUNPOD_ATTACH": "NO",
        }
        result = self.run_script("upload_runpod.sh", CONFIRM_RUN="NO", **env)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.event_list(), [])
        result = self.run_script("upload_runpod.sh", CONFIRM_RUN="YES", RUNPOD_AUTO_STOP="YES", **env)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CONFIRM_RUN=YES bash", self.event_list()[-1])
        self.assertIn("self-reset train YES", self.event_list()[-1])
        self.assertIn("The pod will be asked to stop", result.stdout)

    def test_uploader_only_disables_auto_stop_by_default_for_preflight(self):
        self.write_script("ssh", 'echo "ssh:$*" >> "$EVENTS"\n', directory=self.root)
        self.write_script("rsync", "exit 0\n", directory=self.root)
        for mode, auto_stop in (("preflight", "NO"), ("download", "YES"), ("train", "YES")):
            with self.subTest(mode=mode):
                result = self.run_script(
                    "upload_runpod.sh", RUNPOD_SSH_HOST="pod",
                    RUNPOD_MODE=mode, RUNPOD_ATTACH="NO", CONFIRM_RUN="YES",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"self-reset {mode} {auto_stop}", self.event_list()[-1])

    def test_trial_options_reach_tmux_worker_after_upload(self):
        baseline_log = "/workspace/previous run/console.log"
        host_alias = "[157.66.254.39]:11332"
        self.write_script("ssh", 'printf "%s\\n" "$@" >> "$EVENTS"\n', directory=self.root)
        self.write_script("rsync", 'printf "%s\\n" "$@" >> "$EVENTS"\n', directory=self.root)
        result = self.run_script(
            "upload_runpod.sh", RUNPOD_SSH_HOST="root@pod", RUNPOD_SSH_PORT="12402",
            RUNPOD_SSH_HOST_KEY_ALIAS=host_alias, RUNPOD_MODE="train", RUNPOD_ATTACH="NO",
            CONFIRM_RUN="YES", RUNPOD_TRAIN_TRIAL="YES", REUSE_BASELINE_LOG=baseline_log,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        events = self.event_list()
        self.assertIn(
            f"ssh -p 12402 -o StrictHostKeyChecking=yes -o HostKeyAlias={host_alias}", events,
        )
        self.assertNotIn("-avR", events)
        self.assertLess(events.index("-rlt"), len(events) - 1)
        remote_command = shlex.split(events[-1])
        self.assertEqual(remote_command[:3], [
            "RUNPOD_TRAIN_TRIAL=YES", f"REUSE_BASELINE_LOG={baseline_log}", "CONFIRM_RUN=YES",
        ])

        self.events.unlink()
        self.write_script(
            "tmux",
            '[[ "$1" == new-session ]] || exit 1\nprintf "%s\\n" "$@" > "$EVENTS"\n',
            directory=self.root,
        )
        result = self.run_script(
            "runpod_tmux.sh", *remote_command[-3:],
            **dict(assignment.split("=", 1) for assignment in remote_command[:3]),
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        worker_command = shlex.split(self.event_list()[-1])
        self.assertEqual(worker_command[:2], remote_command[:2])
        self.assertEqual(worker_command[-4:], ["--worker", "train", "YES", "YES"])

    def test_direct_training_and_preflight_fail_without_stopping(self):
        self.write_script("runpod_util.sh", "runpod_ensure_environment() { return 7; }\n")
        result = self.run_script("runpod.sh", CONFIRM_RUN="YES", RUNPOD_AUTO_STOP="NO")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(self.event_list(), [])
        result = self.run_script("runpod.sh", CONFIRM_RUN="NO", RUNPOD_AUTO_STOP="YES")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(self.event_list(), [])

    def test_direct_training_stops_on_failure(self):
        self.write_script("runpod_util.sh", "runpod_ensure_environment() { return 7; }\n")
        result = self.run_script("runpod.sh", CONFIRM_RUN="YES")
        self.assertEqual(result.returncode, 7)
        self.assertEqual(self.event_list(), ["--check", "--stop"])

    def stub_training_environment(self):
        (self.scripts / "models").mkdir()
        self.write_script("models/qwen3-8B.sh", "MODEL_ARGS=(--num-layers 36)\n")
        self.write_script(
            "runpod_util.sh",
            'runpod_ensure_environment() { RUNPOD_PYTHON="$STUBS/python"; RUNPOD_RAY="$STUBS/ray"; }\n'
            'runpod_ensure_data() { TRAIN_DATA="$EVENTS"; EVAL_WILDJAILBREAK="$EVENTS"; '
            'EVAL_XSTEST="$EVENTS"; touch "$EVENTS"; }\n'
            'runpod_set_model_paths() { REF_LOAD="$STUBS/ref"; HF_CHECKPOINT="$REF_LOAD"; '
            'SAFETY_STREAM_MODEL_PATH=stream; SAFETY_GUARD_MODEL_PATH=gen; '
            'LLAMA_GUARD_MODEL_PATH=llama; }\n'
            'runpod_ensure_hf_models() { :; }\n'
            'runpod_create_reference_checkpoint() { mkdir -p "$REF_LOAD"; '
            'touch "$REF_LOAD/latest_checkpointed_iteration.txt"; }\n',
        )
        self.write_script(
            "python",
            'if [[ "$1" == "-c" ]]; then exec "$REAL_PYTHON" "$@"; fi\n'
            'input="$(cat)"; if [[ "$input" == *"print(count)"* ]]; then echo 1; fi\n',
            self.root,
        )
        self.env.update(
            CONFIRM_RUN="YES",
            RUNPOD_AUTO_STOP="NO",
            HF_HOME=str(self.root / "hf"),
            MEGATRON_PATH=str(self.root),
            MODEL_CONFIG="qwen3-8B",
            NUM_EPOCHS="",
            RESUME_FROM="",
            RAY_TMPDIR=str(self.root / "ray-state"),
            STUBS=str(self.root),
            REAL_PYTHON=sys.executable,
        )

    def test_ray_socket_fits_and_startup_failure_logs_are_archived(self):
        self.stub_training_environment()
        self.write_script(
            "curl",
            '''[[ "$*" == */health ]] && echo '{"status":"ok","content_evaluator":true,"gen_label_evaluator":true}'\n''',
            self.root,
        )
        self.write_script(
            "ray",
            'while [[ "$1" != "--temp-dir" ]]; do shift; done\n'
            '[[ "$2" == "$RAY_TMPDIR" ]] || exit 8\n'
            'printf "%s\\n" "$2" >> "$EVENTS"\n'
            'socket="$2/session_2026-11-17_22-40-20_347831_4863/sockets/plasma_store"\n'
            '[[ "$(printf "%s" "$socket" | wc -c)" -le 107 ]] || exit 9\n'
            'mkdir -p "$2/session_latest/logs"\n'
            'echo "startup failure" > "$2/session_latest/logs/raylet.err"\n'
            'exit 7\n',
            self.root,
        )
        log_dir = self.root / "persistent" / "logs"
        result = self.run_script(
            "runpod.sh",
            CONFIRM_RUN="YES",
            RAY_TMPDIR="",
            RUN_LOG_DIR=str(log_dir),
            SAVE_DIR=str(self.root / ("checkpoint-" + "x" * 60)),
            HF_HOME=str(self.root / "hf"),
            MEGATRON_PATH=str(self.root),
            MODEL_CONFIG="qwen3-8B",
            RESUME_FROM="",
            STUBS=str(self.root),
        )
        ray_dir, = self.event_list()
        if Path(ray_dir).parent == Path("/tmp") and Path(ray_dir).name.startswith("ray."):
            self.addCleanup(shutil.rmtree, ray_dir)
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        self.assertEqual((log_dir / "ray" / "raylet.err").read_text(), "startup failure\n")

    def test_trial_submits_three_rollouts_with_bounded_generation_and_checkpoints(self):
        self.stub_training_environment()
        self.write_script(
            "curl",
            'case "$*" in\n'
            '  */health) [[ -f "$STUBS/guard" ]] && cat "$STUBS/guard" ;;\n'
            '  */api/version) [[ -d "$STUBS/ray-state/session_latest/logs" ]] ;;\n'
            '  */api/jobs/) [[ -d "$STUBS/ray-state/session_latest/logs" ]] ;;\n'
            'esac\n',
            self.root,
        )
        self.write_script(
            "start_guard_service.sh",
            '[[ "$GUARD_DEVICE" == cpu ]] || exit 8\n'
            '''echo '{"status":"ok","content_evaluator":true,"gen_label_evaluator":true,"cuda_hidden":true}' > "$STUBS/guard"\n''',
        )
        self.write_script(
            "ray",
            'if [[ "$1" == start ]]; then mkdir -p "$RAY_TMPDIR/session_latest/logs"; '
            'else printf "%s\\n" "$@" > "$EVENTS"; fi\n',
            self.root,
        )

        result = self.run_script("runpod.sh", RUNPOD_TRAIN_TRIAL="YES")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = self.event_list()
        for flag, value in {
            "--num-rollout": "3",
            "--rollout-batch-size": "4",
            "--n-samples-per-prompt": "4",
            "--global-batch-size": "16",
            "--rollout-max-response-len": "512",
            "--eval-max-response-len": "512",
            "--max-tokens-per-gpu": "2048",
            "--sglang-max-total-tokens": "4096",
            "--sglang-mem-fraction-static": "0.45",
            "--save-interval": "1",
        }.items():
            self.assertIn(flag, args, result.stdout + result.stderr)
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertNotIn("--no-save", args)
        self.assertIn("--no-offload-train", args)
        self.assertIn("--optimizer-cpu-offload", args)

        self.write_script(
            "ray",
            '[[ "$1" == job && "$2" == submit ]] || exit 8\n'
            'printf "%s\\n" "$@" >> "$EVENTS"\n',
            self.root,
        )
        self.events.unlink()
        baseline_log = str(self.root / "previous run" / "console.log")
        result = self.run_script(
            "runpod.sh", RUNPOD_TRAIN_TRIAL="YES", RUNPOD_REUSE_RAY="YES", RUNPOD_AUTO_STOP="NO",
            REUSE_BASELINE_LOG=baseline_log,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--address=http://127.0.0.1:8266", self.event_list())
        args = self.event_list()
        self.assertEqual(args[args.index("--reuse-baseline-log") + 1], baseline_log)
        self.assertEqual(args[args.index("--eval-interval") + 1], "5")
        self.assertEqual(args[:2], ["job", "submit"])
        self.assertNotIn("--check", args)
        self.assertNotIn("--stop", args)

        (self.root / "guard").write_text(
            '{"status":"ok","content_evaluator":true,"gen_label_evaluator":true,"cuda_hidden":false}'
        )
        self.events.unlink()
        result = self.run_script("runpod.sh", RUNPOD_TRAIN_TRIAL="YES")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("The trial requires a CPU guard", result.stdout + result.stderr)
        self.assertEqual(self.event_list(), [])


if __name__ == "__main__":
    unittest.main()
