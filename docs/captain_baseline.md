# CaPtain baseline before changing Glimmer

This experiment uses the authors' code directly, outside our `PaperEnv`. It does
not train Muse, run our PPO, fit our experimental critic, change rewards, or apply
object forces. Existing runs and checkpoints remain available.

Sources: [paper §3.1, Appendix A–C](https://arxiv.org/html/2610.40799v1),
[pipeline](https://github.com/waltstephen/CaPtain),
[installation](https://github.com/waltstephen/CaPtain-Harness/blob/de38d6d63b710caf356320a3c046b72b6d299514/INSTALL.md).

## What changes from our current setup

| Current Glimmer experiment | Released CaPtain pipeline |
|---|---|
| One direct `move_to`, `done`, or `give_up` call | One `python` tool; persistent variables, helpers, loops, multiple robot calls per request |
| RGB images and robot state | RGB-D, camera calibration, SAM 3 masks, Molmo2 pointing, pixel back-projection |
| Coordinates inferred largely by the policy | Image-grounded detections and measured object/destination geometry |
| Bounded interpolated motion | Guarded Cartesian motion, IK, force stops, recovery, motion previews |
| Basic tool results | Structured receipts, fresh measurements, completion gate |
| Our Astra import and QLoRA PPO | In-harness SFT, then the authors' physical critic and SAO-style RL |

The extra perception models supply capabilities; this is not evidence that a
different tool schema alone improves the policy's intrinsic vision. The paper
does not isolate that effect with a `move_to`-only ablation. No hidden object
poses or success predicates may be substituted for perception outputs.

## First experiment

Evaluate these three models using **the same unchanged harness and task list**:

1. `Qwen/Qwen3.5-4B`: base model, zero-shot in this harness.
2. `YijiaFan/CaPtain-4B-SFT`: released SFT baseline.
3. `YijiaFan/CaPtain-4B`: released RL baseline.

Start with LIBERO-10 task 0, official evaluation states 11–16 (six episodes per
model). This is a task-specific smoke comparison, not a reproduction of the
paper's aggregate success rate. Repeat over all 240 episodes for that comparison.
These states become evaluation-only for this new experiment; do not reuse them
for new training. They were not necessarily held out from our older experiments,
so old Glimmer checkpoints cannot be described as clean held-out comparisons.

The downloaded pipeline is pinned to `4bb86546196b80aaa2fa4bd0f9fa697aa7580e8e`;
its harness submodule is pinned to `de38d6d63b710caf356320a3c046b72b6d299514`.
The launcher rejects changed tracked files or a different revision. It saves task
hashes, model source declarations, endpoint model metadata, and launch settings.
It **does not attest the endpoint's actual weight hashes**; pin/download model
revisions and retain their hashes separately for a reproducible result.

## Prepare on cs94085

After these repository changes have been pushed and pulled:

```bash
cd /data3/may/muse-glimmer-ppo
python3 scripts/captain_baseline.py prepare
```

This downloads source only, into ignored `data/captain/upstream`. It creates
`data/captain/task0_official/tasks.tsv` using unmodified rows from the official
task list. It does not install dependencies or start GPU services.

Use separate environments: the old `llm-agent-memory` environment is Python 3.12
and is not the authors' required simulator or sandbox environment.

```bash
cd /data3/may/muse-glimmer-ppo/data/captain/upstream/harness
bash scripts/setup_third_party.sh --capx

python3.10 -m venv .venv-libero
.venv-libero/bin/pip install -r requirements-libero.txt
.venv-libero/bin/pip install -e third_party/libero --no-deps
.venv-libero/bin/pip install -e . --no-deps

# Must be Python 3.11.16; check the interpreter before creating this environment.
python3.11 --version
python3.11 -m venv .venv-student
.venv-student/bin/pip install -r requirements-student.txt
export STUDENT_PYTHON="$PWD/.venv-student/bin/python"
```

If Python 3.10, Python 3.11.16, or `bwrap >= 0.5` are missing, install those first.
Keep these interpreters separate. The policy and Molmo services use a separate
vLLM environment; the authors specify vLLM 0.29.0 / transformers 5.17.0 for
evaluation. SAM 3 needs a separate CaP-X/SAM environment; follow the pinned
`INSTALL.md` and `third_party/cap-x/README.md`. Do not install its conflicting
simulator dependencies into `.venv-libero`.

SAM 3 weights require a Hugging Face account with access to `facebook/sam3`.
Set `HF_TOKEN` in the terminal if needed; never paste or commit it. Download the
models with the authors' scripts. Molmo's revision is pinned. The SAM script
prints a hash comparison but only warns on mismatch: resolve any mismatch before
calling the test an exact baseline.

```bash
export CAPTAIN_MODELS_DIR=/data3/may/captain-models
bash scripts/download_models.sh
bash scripts/models/download_policy.sh 4b-sft
bash scripts/models/download_policy.sh 4b
```

## Start perception and one policy

Set `TOOLS_PYTHON` to the installed CaP-X/SAM Python and `VLLM_BIN` to the vLLM
executable. The commands below require those paths to be set; they deliberately
do not guess another project's Python environment.

Use GPU 1 for the policy, GPU 2 for SAM, GPU 3 for Molmo and rendering. This
avoids GPU 0's existing job. This placement is an initial test on the 48 GB Ada
cards, not a verified memory estimate. Start one episode at a time. The policy
script retains the authors' context and serving settings; if they fail to fit,
report that failure before reducing context or changing precision.

```bash
cd /data3/may/muse-glimmer-ppo/data/captain/upstream/harness
: "${TOOLS_PYTHON:?Set TOOLS_PYTHON to the CaP-X/SAM interpreter}"
: "${VLLM_BIN:?Set VLLM_BIN to the vLLM executable}"
export SAM3_CHECKPOINT="$CAPTAIN_MODELS_DIR/sam3/sam3.pt"
export MOLMO_MODEL="$CAPTAIN_MODELS_DIR/Molmo2-8B"
export SAM3_GPU=2 MOLMO_GPU=3 POLICY_GPU=1 EGL_DEVICE=3
bash services/start_perception.sh
bash services/serve_policy.sh "$CAPTAIN_MODELS_DIR/CaPtain-4B-SFT" 8600
```

These scripts detach their services. Follow startup with
`tail -f logs/services/policy_8600.log`; SAM and Molmo logs are in that directory
too. **Do not start a second policy on port 8600 until the first is stopped.**
The launch scripts print their PIDs: stop only the specific policy PID you
started when switching models. Keep perception running. Do not use broad
`pkill` commands on this shared machine.

## Check, run, and inspect

Keep the exported variables in the same terminal, or re-export them in a tmux
session. The readiness check validates interpreter versions, imports, `bwrap`
presence, policy/Molmo model endpoints and SAM route availability. It does not
prove segmentation inference, bubblewrap permissions or EGL work; the first
episode is the integration test.

```bash
cd /data3/may/muse-glimmer-ppo
python3 scripts/captain_baseline.py check

# Optional: prints the six upstream commands; launches no episodes.
python3 -u scripts/captain_baseline.py run \
  --model-source YijiaFan/CaPtain-4B-SFT \
  --out runs/captain_sft_task0_dry --dry-run

# Run in tmux so a lost SSH connection does not interrupt evaluation.
python3 -u scripts/captain_baseline.py run \
  --model-source YijiaFan/CaPtain-4B-SFT \
  --out runs/captain_sft_task0 --parallel 1
```

This calls their `eval/run_jobs.py`, which calls their unchanged
`eval/run_episode.sh`. Settings include temperature 1, top-p 1, thinking enabled,
60 requests, 150 physical actions, 1080-second wall budget, 64k context, 12 images,
and the authors' prompts, Python schema, compaction and completion gate.

The official runner writes per-episode logs instead of streaming every request
to this terminal. From another terminal:

```bash
cd /data3/may/muse-glimmer-ppo
python3 scripts/captain_baseline.py summary --out runs/captain_sft_task0
tail -f runs/captain_sft_task0/logs/libero_30.log
```

The task 0 / seed 11 official row has ID 30 at the pinned revision. The summary
reports completed episodes, host success, strict success (host success plus
`finish()`), termination reasons, and missing/infrastructure cases. Infrastructure
errors are not counted as robot-policy failures. Our existing PPO viewer does
not yet read this upstream output format.

Next stop that policy's PID, serve `YijiaFan/CaPtain-4B` with the same command,
and evaluate into `runs/captain_rl_task0` with `--model-source` changed accordingly.
Then repeat for `Qwen/Qwen3.5-4B` into `runs/captain_base_task0`. Keep the task plan,
perception and settings unchanged. Model sources may also be downloaded local
paths; record the source revision/hashes alongside the run.

For the full official evaluation, prepare a **separate plan** and pass it to all
subsequent `check`, `run`, and `summary` commands:

```bash
python3 scripts/captain_baseline.py prepare --full --plan data/captain/official240
```

## Training and adding our changes

Evaluating released weights verifies the inference pipeline; it does not reproduce
training. The exact training entry points are the pinned upstream
`sft/train/launch_sft.sh` with `sft/configs/captain-4b-sft.yaml`, and
`rl/scripts/run_stream.sh rl/configs/captain-4b.json`. Follow their SFT/RL READMEs
for released data, critic warm start, start bank, packing, loss normalization and
environments. Do not substitute our PPO loop and call it an exact reproduction.

The released SFT run used 16 H100-80GB GPUs. RL provides 16- and 8-GPU layouts;
there is no validated four-Ada recipe in the pinned release. Adapting to our
hardware is a separate experiment and must be recorded as such.

After the baseline runs, add one change per comparison: first an adapter that
serves Glimmer in this same Python protocol, then in-harness SFT using this exact
input/output format, then RL changes. Our existing move-only Astra examples are
not the authors' in-harness Python trajectories. Keep policy, task states, budgets
and perception fixed when comparing a reward, critic, or assistance change.
Compare against the SFT/RL baselines as well as zero-shot; a zero-shot gain alone
does not establish improvement over the paper's trained baseline.
