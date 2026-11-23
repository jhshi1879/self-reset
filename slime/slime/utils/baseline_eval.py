import ast
import re
from argparse import Namespace
from pathlib import Path


_BASELINE_FIELDS = (
    "eval_datasets", "n_samples_per_eval_prompt", "eval_max_response_len",
    "eval_min_new_tokens", "eval_temperature", "eval_top_p", "eval_top_k",
    "eval_repetition_penalty", "eval_function_path", "eval_reward_key",
    "eval_input_key", "eval_label_key", "eval_tool_key", "rollout_stop",
    "rollout_stop_token_ids", "rollout_skip_special_tokens", "rollout_max_context_len",
    "seed", "rollout_seed", "sglang_enable_deterministic_inference", "apply_chat_template",
    "hf_checkpoint", "load", "ckpt_step", "custom_rm_path", "rm_type", "use_llama_guard",
    "safety_stream_model_path", "safety_guard_model_path", "llama_guard_model_path",
)
_ARGUMENT_LINE = re.compile(r"^\s+(\w+) +\.{3,} (.*)$")
_TERMINAL_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_INITIAL_METRICS = re.compile(r"^(?:\(RolloutManager pid=\d+\)\s+)?eval 0: (\{.*\})$")
_CHECKPOINT_SAVE = re.compile(r"^(?:\(MegatronTrainRayActor pid=\d+\)\s+)?saving checkpoint at iteration\s+\d+")


def reuse_initial_eval(args: Namespace) -> bool:
    if not args.reuse_baseline_log:
        return False
    if args.evaluation_only or args.num_rollout == 0 or args.start_rollout_id != 0:
        raise ValueError("--reuse-baseline-log requires a fresh training run.")

    path = Path(args.reuse_baseline_log)
    expected = {key: str(getattr(args, key)) for key in _BASELINE_FIELDS}
    logged = {}
    with path.open(encoding="utf-8") as log:
        for raw_line in log:
            line = _TERMINAL_ESCAPE.sub("", raw_line).rstrip()
            if line.startswith(("Reused initial evaluation from ", "last_epoch_id:")) or _CHECKPOINT_SAVE.match(line):
                break
            argument = _ARGUMENT_LINE.match(line)
            if argument and argument[1] in expected:
                logged[argument[1]] = argument[2].strip()
            match = _INITIAL_METRICS.match(line)
            if not match:
                continue
            mismatches = [key for key, value in expected.items() if logged.get(key) != value]
            if mismatches:
                raise ValueError(f"Cannot reuse initial evaluation from {path}: missing or different "
                                 f"settings: {', '.join(mismatches)}")
            metrics = ast.literal_eval(match[1])
            if not isinstance(metrics, dict) or not all(
                type(metrics.get(f"eval/{dataset.name}")) in (int, float)
                for dataset in args.eval_datasets
            ):
                break
            print(f"Reused initial evaluation from {path}: {metrics}", flush=True)
            return True
    raise ValueError(f"No complete initial evaluation found in {path}.")
