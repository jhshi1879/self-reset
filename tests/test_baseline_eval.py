import contextlib
import importlib.util
import io
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest


_PATH = Path(__file__).resolve().parents[1] / "slime/slime/utils/baseline_eval.py"
_SPEC = importlib.util.spec_from_file_location("baseline_eval", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
baseline_eval = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(baseline_eval)


class BaselineEvalTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.log = Path(temp.name) / "console.log"
        self.args = Namespace(**dict.fromkeys(baseline_eval._BASELINE_FIELDS))
        vars(self.args).update(
            reuse_baseline_log=str(self.log),
            evaluation_only=False,
            num_rollout=3,
            start_rollout_id=0,
            eval_max_response_len=512,
            eval_datasets=[Namespace(name="wildjailbreak"), Namespace(name="XSTest")],
        )

    def write_log(self, scores, prefix="(RolloutManager pid=123) ", suffix=""):
        settings = "".join(
            f"  {field} ................ {getattr(self.args, field)}\n"
            for field in baseline_eval._BASELINE_FIELDS
        )
        self.log.write_text(f"{settings}{prefix}eval 0: {scores}{suffix}\n")

    def test_matching_log_reuses_both_scores(self):
        self.write_log({"eval/wildjailbreak": 0.55, "eval/XSTest": 1.0})
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertTrue(baseline_eval.reuse_initial_eval(self.args))
        self.assertIn("'eval/wildjailbreak': 0.55, 'eval/XSTest': 1.0", output.getvalue())

    def test_changed_token_limit_refuses_reuse(self):
        self.write_log({"eval/wildjailbreak": 0.55, "eval/XSTest": 1.0})
        self.args.eval_max_response_len = 1024
        with self.assertRaisesRegex(ValueError, "eval_max_response_len"):
            baseline_eval.reuse_initial_eval(self.args)

    def test_terminal_formatting_does_not_hide_baseline(self):
        self.write_log(
            {"eval/wildjailbreak": 0.55, "eval/XSTest": 1.0},
            prefix="\x1b[36m(RolloutManager pid=42672)\x1b[0m ",
            suffix="\x1b[0m  ",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(baseline_eval.reuse_initial_eval(self.args))

    def test_post_training_eval_zero_is_not_a_baseline(self):
        self.write_log(
            {"eval/wildjailbreak": 0.55, "eval/XSTest": 1.0},
            prefix="(MegatronTrainRayActor pid=43328) saving checkpoint at iteration       0\n"
                   "(RolloutManager pid=42672) ",
        )
        with self.assertRaisesRegex(ValueError, "No complete initial evaluation"):
            baseline_eval.reuse_initial_eval(self.args)

    def test_missing_dataset_score_refuses_reuse(self):
        self.write_log({"eval/wildjailbreak": 0.55})
        with self.assertRaisesRegex(ValueError, "No complete initial evaluation"):
            baseline_eval.reuse_initial_eval(self.args)

    def test_no_option_keeps_initial_eval(self):
        self.args.reuse_baseline_log = None
        self.assertFalse(baseline_eval.reuse_initial_eval(self.args))
