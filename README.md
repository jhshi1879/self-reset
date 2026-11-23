# Autonomous recovery after unsafe reasoning

This folder collects training and evaluation files relevant to Section 5.3
and Figure 3 of [Self-ReSET](https://arxiv.org/abs/2605.08936). Originals
remain in place.

The experiment compares base, DAPO, RECAP, and Self-ReSET models across
three model families on WildJailbreak and Fortress. Its recovery rate is
the number of generated responses judged safe among those whose reasoning
trajectories were judged unsafe, divided by the number of unsafe reasoning
trajectories. Appendix C describes Qwen3Guard-Stream monitoring with two
consecutive non-safe token labels; Appendix A.4 identifies LlamaGuard-3 as
the final safety judge.

## Included files

- `data/evaluation/safety/wildjailbreak.jsonl` and
  `data/evaluation/safety/fortress_adversarial.jsonl`: the two evaluation
  datasets (250 and 500 prompts respectively). Each `prompt` is a list of
  chat messages, not a plain string.
- `slime/slime/rollout/Guardforward.py`: streaming and final-response guard
  implementations.
- `slime/slime/rollout/partialrollout.py`: generation, evaluation, and
  consecutive non-safe token detection code.
- `slime/slime/rollout/rm_hub/safety_rm.py`: final-response reward and
  safety-label integration.
- `slime/slime/utils/types.py` and package initializers: supporting source
  files. `slime/LICENSE` retains the source license.
- `data/training/vanilla_filter_balanced_1500.jsonl`: balanced training data.
- `slime/train_epoch.py`: training entry point.
- `slime/scripts/example.sh`: original training launcher.
- `slime/scripts/start_guard_service.sh`: guard service launcher.
- `slime/scripts/models/`: architecture configurations for the three model
  families in the paper.
- `slime/slime/` and `slime/slime_plugins/`: framework runtime and plugins.
- `slime/tools/convert_hf_to_torch_dist.py`: checkpoint conversion.
- `slime/setup.py`, `slime/pyproject.toml`, and `slime/requirements.txt`:
  Python package metadata and declared dependencies.

## Training prerequisites

The copied scripts are the original examples, not ready-to-run experiment
configuration. Before running them, provide the base and guard checkpoints,
configure their paths, the data paths and the Megatron-LM path, and select the
appropriate model configuration. The example uses four training epochs; the
paper reports three for Self-ReSET. Its evaluation settings also differ from
the paper.

**Do not run `slime/scripts/example.sh` unchanged on a shared machine.** It
starts by force-killing processes by name and stopping Ray; remove or replace
those commands for your environment. It also contains placeholder paths.

The Python requirements file is not a complete GPU environment: PyTorch,
CUDA, SGLang, Megatron-LM, and their compatible dependencies must be provided
separately. Checkpoints are not included. The original repository recommends
eight NVIDIA GPUs.

## RunPod single-pod launcher

Create a Linux x86_64 GPU pod using the image
`slimerl/slime:nightly-dev-20260430b` and expose TCP port 22. This image
contains the prebuilt GPU stack; changing the launcher cannot change the image
of an existing pod. The image is about 25 GB compressed, so allow additional
container storage for its unpacked layers. Preserve the `/workspace` volume
when creating a replacement pod if it contains data you need.

The image includes OpenSSH but does not start an SSH server by default. Set
the pod environment variable `PUBLIC_KEY` to your SSH **public** key (for
example, the contents of `~/.ssh/id_ed25519.pub` on your computer). Registering
the key in RunPod credentials alone does not necessarily make `PUBLIC_KEY`
available to this custom image. Never upload or paste the private key.
Configure this pod start command and use the direct SSH connection from the
pod's Connect tab:

```bash
bash -c 'set -e; echo "Self-ReSET startup reached"; if [ -z "$PUBLIC_KEY" ]; then echo "ERROR: PUBLIC_KEY is missing" >&2; exit 1; fi; install -d -m 700 /root/.ssh /run/sshd; printf "%s\n" "$PUBLIC_KEY" >> /root/.ssh/authorized_keys; chmod 600 /root/.ssh/authorized_keys; ssh-keygen -A; exec /usr/sbin/sshd -D -e'
```

Once the pod is running and SSH access is configured, upload the local
replication files and start preflight in tmux from your computer:

```bash
RUNPOD_SSH_HOST=root@YOUR_POD_HOST RUNPOD_SSH_PORT=YOUR_SSH_PORT \
  bash replication/slime/scripts/upload_runpod.sh
```

Use the direct SSH host and port shown by RunPod, or an SSH config alias
for `RUNPOD_SSH_HOST`. SSH authentication and host-key verification must be
configured normally, and `rsync` must be installed locally and on the pod.
The upload targets `/workspace/Self-ReSET/replication` by default; override
its parent with `RUNPOD_PROJECT_DIR`. It copies source and data files, excludes
hidden files including `.env` and symlinks, and does not start training.
It updates matching files on the pod without deleting remote files. Do not put
secrets in source or data files.

The script starts a detached `self-reset` tmux session over SSH, installing
tmux with `apt-get` if needed and possible. The session starts or reuses the
guard service after preparing the reference checkpoint, then runs the
remaining preflight checks. Its output is also saved
under `SAVE_DIR/logs/tmux-*.log`. On an interactive terminal the script attaches
automatically; set `RUNPOD_ATTACH=NO` to leave it detached. Detach with
Ctrl-b then d, and reconnect with the SSH command printed by the uploader.
Auto-stop is off by default for preflight, so the pod and tmux session remain running after
preflight. Once preflight passes, run the displayed training command inside
tmux. Alternatively, use the uploader in train mode after exiting the
preflight session (or choose a different `RUNPOD_TMUX_SESSION`):

```bash
RUNPOD_MODE=train CONFIRM_RUN=YES RUNPOD_SSH_HOST=root@YOUR_POD_HOST RUNPOD_SSH_PORT=YOUR_SSH_PORT \
  bash replication/slime/scripts/upload_runpod.sh
```

Train mode starts or reuses the guard service and requests a pod stop on
success or failure by default. Set `RUNPOD_AUTO_STOP=NO` to leave it running.
Preflight can opt in with `RUNPOD_AUTO_STOP=YES`. Auto-stop checks API access
before work begins, but stop permission is only verified when requested;
check the Runpod console and stop the pod manually if the request fails.
Closing your laptop does not stop tmux; stopping or restarting the pod does.

The pod secret `HF_TOKEN` must be available to the SSH session. Some images do
not export pod environment variables to SSH; the tmux worker warns when
`HF_TOKEN` is absent. Authenticate on the pod or configure the image to expose
the secret to SSH rather than copying the local `.env`.

### CPU-only downloads

Stop the GPU pod before starting downloads on a CPU pod. Attach the same network
volume at `/workspace` when creating the CPU pod, and use an image with Python
3.10+, `curl`, `rsync`, and working direct SSH. Keep the existing Hugging Face
login under `/workspace/.cache/huggingface`, or authenticate again if needed.
From your computer, use the CPU pod's own SSH host and port:

```bash
RUNPOD_MODE=download RUNPOD_SSH_HOST=root@YOUR_CPU_POD_HOST RUNPOD_SSH_PORT=YOUR_CPU_SSH_PORT \
  bash replication/slime/scripts/upload_runpod.sh
```

The detached tmux session downloads the base model, all three guards and the
datasets, reusing existing files. It does not initialize CUDA, start guard
services, convert checkpoints or train. If `huggingface_hub` is missing, it
installs that download dependency into a separate persistent virtual
environment; Python venv and pip must be available. The session reports
completion or errors to the persistent tmux log. By default the worker requests
a CPU pod stop on either success or failure; set `RUNPOD_AUTO_STOP=NO` to opt out.
When auto-stop is enabled, the worker checks Runpod API access before downloading. It needs the pod ID and
pod-scoped API key, read from its
environment or the container's PID 1 environment, without logging the key.
Stop permission is not guaranteed by the read-only access check. If the stop
request fails, an error is written to the log and the pod must be stopped
manually; verify its state in the Runpod console. Do not run two downloaders
against the same model directories. After completion, attach the same network
volume to a GPU pod for conversion and normal preflight.

`slime/scripts/runpod.sh` and the guard launcher validate the prebuilt Python,
GPU and required imports before downloading models. They do not install or
compile GPU dependencies; an incomplete or incompatible environment fails
with its error. The defaults are `/usr/bin/python3.12`, `/usr/local/bin/ray`
and `/root/Megatron-LM` from the image. An old environment under `/workspace`
is not selected automatically. `MEGATRON_PATH` identifies the Megatron-LM
source checkout; set it only to use a different existing checkout. Advanced
users can explicitly set `RUNPOD_VENV`, `RUNPOD_PYTHON`, and `RUNPOD_RAY`.
The image's compatibility with this copied project has not yet been verified
on a GPU pod.

The guard launcher creates a separate system-site-packages virtual environment
at `/workspace/.venvs/self-reset-guard` and installs the pinned guard
Transformers version there. The guard imports shared sample types without
loading the training argument parser or SGLang. It reuses the image's PyTorch
and leaves the image's Transformers and SGLang packages unchanged. Python venv, pip, and
network access for the guard dependencies are required. The environment is
reused only with a compatible base Python; use a fresh `GUARD_VENV` path if
the image or Python changes. Starting the guard service still requires GPU
memory for the guard models.
The default guard-based reward does not need `gpu_client_v2`. The optional
GPT-4o judge still requires that separate client and `FILE_API_PROXY_PATH`.

The training launcher places its three required JSONL datasets under
`/workspace/data/self-reset` by default, reusing valid local repository copies
or downloading missing files from a pinned Self-ReSET commit. It verifies
SHA-256 checksums before use. Override the destination with
`RUNPOD_DATA_DIR`, or provide existing custom files with `TRAIN_DATA`,
`EVAL_WILDJAILBREAK`, and `EVAL_XSTEST`.

The default model configuration is `qwen3-8B`, using `Qwen/Qwen3-8B` rather
than the DeepSeek distilled 7B model. Model snapshots are stored under
`/workspace/models`; the launchers download missing base and guard models from
Hugging Face, then create the Megatron reference checkpoint from the base
model if `REF_LOAD` is missing. Override repository IDs with
`HF_BASE_MODEL_ID`, `QWEN3GUARD_STREAM_MODEL_ID`, `QWEN3GUARD_GEN_MODEL_ID`,
and `LLAMA_GUARD_MODEL_ID`; override local paths with `HF_CHECKPOINT`,
`SAFETY_STREAM_MODEL_PATH`, `SAFETY_GUARD_MODEL_PATH`, `LLAMA_GUARD_MODEL_PATH`,
and `REF_LOAD`.

Default local paths:

- Base HF model: `/workspace/models/Qwen3-8B`
- Converted Megatron reference: `/workspace/checkpoints/Qwen3-8B_torch_dist`
- Qwen3Guard Stream: `/workspace/models/Qwen3Guard-Stream-8B`
- Qwen3Guard Gen: `/workspace/models/Qwen3Guard-Gen-8B`
- Llama Guard: `/workspace/models/Llama-Guard-3-8B`
- Training outputs: `/workspace/checkpoints/Self-RESET`

The Llama Guard repository is gated. Sign in to Hugging Face, open
[Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B), and
accept/request access to Meta's license. After access is granted, create a
read token in [Hugging Face settings](https://huggingface.co/settings/tokens)
and add it to the pod as the secret environment variable `HF_TOKEN`. You can
also export `HF_HOME=/workspace/.cache/huggingface`, run `hf auth login` after
setup, and paste the token at the prompt. Do not put the token in this
repository or in a script.

The downloader fetches safetensors and required tokenizer/configuration files,
not duplicate PyTorch `.bin` weights. It checks available space before
downloading. The four model snapshots plus the converted base checkpoint need
roughly 85 GiB free, including a 10 GiB setup reserve; this excludes the GPU
software image, training checkpoints and logs. A network filesystem may report
the capacity of its shared backing store rather than the volume's quota; check
the RunPod volume quota separately.

The launcher detects the GPUs visible to PyTorch, selects tensor parallel
size 2 when the GPU count is even (otherwise 1), and uses that count for the
Ray actor and colocated rollout. It does not kill existing processes, stop
Ray, or install dependencies beyond the setup described above. It defaults to
**one rollout/training step**
(`NUM_ROLLOUTS=1`); set `NUM_EPOCHS=1` to run a full epoch instead. It retains
only the newest completed checkpoint in `SAVE_DIR` after each save. Console output is written to
`SAVE_DIR/logs/<run-id>/console.log`; Ray runs in a short, pod-local
`/tmp/ray.*` directory to stay within the Unix socket path limit. Its logs
are copied to `SAVE_DIR/logs/<run-id>/ray/` when the launcher exits. If
overriding `RAY_TMPDIR`, use a short absolute path. The guard service helper writes its console to
`SAVE_DIR/logs/guard-<timestamp>.log`.

Training starts only with `CONFIRM_RUN=YES`. GPU detection does not guarantee
that the model, guard service, and batch settings fit in the pod's VRAM; adjust
the settings below for the detected hardware. The launcher starts its own
single-node Ray head on ports 6380/8266 by default and refuses to attach to an
already-running Ray dashboard; choose unused `RAY_GCS_PORT` and
`RAY_DASHBOARD_PORT` values if those ports are occupied.

Before using it:

1. Use the prebuilt image above on a Linux x86_64 GPU pod with a compatible
   NVIDIA driver and working direct SSH.
2. Use a persistent volume with enough space for the GPU software, model
   snapshots, conversion output, checkpoints, and logs.
3. If using Llama Guard, accept its gated-model terms and configure `HF_TOKEN`
   as described above.
4. Start the guard service before training. Its helper defaults to the last GPU
   visible to PyTorch; `GUARD_GPU_ID` or its first argument can select another
   visible GPU.

The launchers automatically check the image environment. You can also run the
validation explicitly to fail early:

```bash
export HF_HOME=/workspace/.cache/huggingface
export MODEL_CONFIG=qwen3-8B

bash replication/slime/scripts/runpod_setup.sh

export SAVE_DIR=/workspace/checkpoints/Self-RESET

RUNPOD_START_GUARD=YES bash replication/slime/scripts/runpod.sh
```

The final command downloads any missing models, converts the reference
checkpoint if needed, and then performs preflight checks without submitting
training. To submit training, rerun it as
`CONFIRM_RUN=YES bash replication/slime/scripts/runpod.sh`. This direct
training command leaves the pod running on success or failure, as does
direct preflight.
Optional settings include `MODEL_CONFIG`, `TENSOR_PARALLEL_SIZE`, `NUM_ROLLOUTS`,
`NUM_EPOCHS`,
`ROLLOUT_BATCH_SIZE`, `N_SAMPLES_PER_PROMPT`, `ROLLOUT_MAX_RESPONSE_LEN`,
`GLOBAL_BATCH_SIZE`, `MAX_TOKENS_PER_GPU`, and `SGLANG_MEM_FRACTION_STATIC`.
The launcher evaluates WildJailbreak and XSTest before training and after the
final rollout; longer runs also evaluate every 5 rollouts. Evaluations use 2
samples per prompt and the original example's generation settings.
Override dataset paths with `EVAL_WILDJAILBREAK` and `EVAL_XSTEST`, or the
cadence with `EVAL_INTERVAL`. The default batch and sequence settings mirror
the original example and may need to be reduced for smaller-memory GPUs. The
existing evaluation loop is not a complete Figure 3 replication, as noted
below.

On the pinned Slime image, the launcher makes SGLang's `/health` endpoint
non-generating. This lets the router probe the worker while its GPU memory is
offloaded. Startup still checks `/health_generate` before offload.

The single-H100 Qwen3-8B smoke run used CPU guards and optimizer offload,
kept the training model on GPU (`--no-offload-train`), and limited SGLang
to `--sglang-mem-fraction-static 0.45 --sglang-max-total-tokens 4096` with
both `--sglang-disable-cuda-graph` and `--sglang-disable-piecewise-cuda-graph`.
This was a Ray job override, not the default launcher configuration. It ran
one rollout (two prompts, two samples each), one optimizer step, and one
final example per evaluation dataset, with generation capped at 128 tokens
and checkpoint saving disabled. It verifies the training pipeline, not
learning quality or the original workload's memory requirements.

### Three-rollout single-H100 trial

From the local project root, sync the code and start the trial in tmux with one command:

```bash
RUNPOD_SSH_HOST=root@YOUR_POD_HOST RUNPOD_SSH_PORT=YOUR_SSH_PORT \
  RUNPOD_MODE=train RUNPOD_TRAIN_TRIAL=YES CONFIRM_RUN=YES \
  bash replication/slime/scripts/upload_runpod.sh
```

You can use an SSH config alias as `RUNPOD_SSH_HOST`. If a migrated pod's host
key has been verified against an existing `known_hosts` entry, set
`RUNPOD_SSH_HOST_KEY_ALIAS` to that entry's host alias.

To reuse the initial evaluation from an earlier fresh run, set
`REUSE_BASELINE_LOG=/absolute/path/on/pod/to/console.log` on the same command line.
The driver checks the logged evaluation settings, model and checkpoint
paths, and scores for all configured datasets before skipping the initial
evaluation. If the check fails, the job exits rather than silently rerunning
the evaluation. Use this only when the datasets and model files at those
paths have not changed. Later evaluations, including the final one, still run.
Use the original run's `console.log`; terminal color codes are ignored, and
metrics logged after checkpoint saving starts are not treated as a baseline.

For an already synced checkout, you can invoke `runpod.sh` directly on the pod
with `RUNPOD_TRAIN_TRIAL=YES CONFIRM_RUN=YES`.

The opt-in trial uses CPU guards and the smoke run's memory settings. It runs
three rollouts of four prompts with four generations per prompt (global batch
16), caps both training and evaluation responses at 512 new tokens, and uses
a trainer token budget of 2,048 per GPU. It saves after each rollout in a
fresh trial directory by default and retains the latest checkpoint. The
launcher leaves the pod running when the trial succeeds or fails.

The trial starts the local CPU guard if needed. If port 8100 already serves an
older guard that does not report whether CUDA is hidden, restart that guard
with `GUARD_DEVICE=cpu bash replication/slime/scripts/start_guard_service.sh`
before launching. If the smoke-test Ray head is still running and idle, set
`RUNPOD_REUSE_RAY=YES RAY_DASHBOARD_PORT=8266` to submit to it without
starting a second head. Changing only the dashboard and GCS ports does not
isolate Ray's internal agent ports. The trial is a capacity check, not a
verified full-epoch configuration.

## Replication status

This is a focused source and data extract, **not a complete Figure 3
replication**. The existing evaluation loop applies reasoning detection only
to a few printed examples and does not compute the Figure 3 conditional
recovery rate.

An executable replication still requires an evaluation driver that generates
complete, unprompted reasoning and final responses for every model and
benchmark, labels the reasoning and final response separately, and
aggregates the conditional rate. It also requires the model checkpoints,
guard checkpoints, GPU environment, and any additional framework
dependencies. The paper reports avg@4 for safety evaluations; its exact
Figure 3 sampling configuration is not separately specified in Section 5.3.
Do not treat results from the existing evaluation loop as Figure 3 results.
