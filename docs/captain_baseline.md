# Glimmer in the CaPtain pipeline

The policy under test is **Muse Glimmer 30B**, throughout. We reuse the authors'
harness and training design, not their Qwen policy. This first stage evaluates
frozen Glimmer; SFT and RL come after its tool interface works. Existing runs and
checkpoints remain available.

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

Keep the Glimmer checkpoint fixed for the first two conditions:

1. **Glimmer + our existing move-only harness:** the current baseline.
2. **The same Glimmer + CaPtain-Harness:** persistent Python, perception,
   geometry, guarded motion and receipts; no new training yet.
3. **Glimmer + in-harness SFT**, followed by **Glimmer + the physical-critic RL
   design**. Compare each stage before adding our reward or assistance changes.

The authors' Qwen checkpoints are optional external references, not required
experiments. Swapping Glimmer for Qwen would confound the model and harness.
Reusing the method with Glimmer is an adaptation, not an exact reproduction of
CaPtain-4B or a guarantee of its reported success rates.

Start with LIBERO-10 task 0, official states 11–16 (six episodes per condition).
Use the same states for the paired old/new harness comparison and record both
sets of budgets: the harnesses have different request/action definitions. This
is an end-to-end harness comparison, not an isolated tool-count ablation.
These states are evaluation-only for new training. Older Glimmer checkpoints may
already have trained on them; do not call those comparisons clean held-out tests.

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
# Keep using the existing Muse Glimmer model weights; no Qwen download.
```

## Start perception and one policy

Set `TOOLS_PYTHON` to the installed CaP-X/SAM Python and `VLLM_BIN` to the vLLM
executable. The commands below require those paths to be set; they deliberately
do not guess another project's Python environment.

The Glimmer serving bridge is `scripts/serve_glimmer_python.py`. It loads the
frozen base Glimmer model, uses its native HF multimodal template, and accepts
CaPtain's `python(input=...)` schema. It preserves every complete ATEM call in
order and matches tool-result IDs. Python source is kept raw, including `<`,
`&`, quotes, and indentation. Thinking examples are not executable calls;
truncated or malformed tool replies forward no code.

The server does not execute Python. CaPtain owns the sandboxed persistent
session, robot tools, state, receipts and images. The model server creates no
LoRA adapter, value head or optimizer. It accepts inline images, never fetches
image URLs, and never silently truncates the prompt. It is bound to localhost.
The server logs request counts, token progress and parse results in the terminal;
`requests.jsonl` retains its raw generated replies and parsed calls.

This is Glimmer with CaPtain's harness, not an exact reproduction of the Qwen
training run. The native template and configurable reasoning differ. NF4 is an
explicit inference setting, not a learned adapter. If a prompt exceeds the
configured context budget, the server reports an error; it does not hide that
change by clipping images or history.

### First check: no robot or perception services needed

The existing Glimmer Python environment can run the bridge. Stop any old process
that **you** started on port 8600 first; do not use a broad process kill.

```bash
cd /data3/may/muse-glimmer-ppo
PPO_PY=/data3/may/miniconda3/envs/llm-agent-memory/bin/python

# Cached processor only: no model weights loaded, no generation, no Python run.
"$PPO_PY" scripts/check_glimmer_python.py --template-only

# In tmux: frozen base Glimmer, two GPUs; GPU 0 remains untouched.
CUDA_VISIBLE_DEVICES=1,2 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
"$PPO_PY" -u scripts/serve_glimmer_python.py \
  --load-in-4bit --port 8600 --output runs/glimmer_python_server
```

Wait for `[glimmer] ready`. From another terminal:

```bash
cd /data3/may/muse-glimmer-ppo
PPO_PY=/data3/may/miniconda3/envs/llm-agent-memory/bin/python
export STUDENT_PYTHON="$PWD/data/captain/upstream/harness/.venv-student/bin/python"
"$PPO_PY" -u scripts/check_glimmer_python.py --output runs/glimmer_python_smoke
```

The live check sends an image, asks Glimmer for Python, executes it in the
upstream bubblewrap sandbox, returns the receipt, and asks it to reuse a variable
in a second request. It expects printed `41` then `42`, without a Python restart.
No robot is moved. A pass checks interface operation; it does not establish
visual grounding or robot success. The sandbox requires the upstream setup and
its student Python above; no unsandboxed fallback is used.

If the model returns malformed calls, the live check stops and the server's
`requests.jsonl` shows the raw reply. If it runs out of memory, keep that failure
as a resource result before changing context/precision. The 30B model's memory
use plus perception has not been validated on cs94085 yet.

Perception can use separate GPUs once the separate environments are installed:

```bash
cd /data3/may/muse-glimmer-ppo/data/captain/upstream/harness
: "${TOOLS_PYTHON:?Set TOOLS_PYTHON to the CaP-X/SAM interpreter}"
: "${VLLM_BIN:?Set VLLM_BIN to the Molmo vLLM executable}"
export SAM3_CHECKPOINT="$CAPTAIN_MODELS_DIR/sam3/sam3.pt"
export MOLMO_MODEL="$CAPTAIN_MODELS_DIR/Molmo2-8B"
# After checking free memory: share GPU 3 for perception; Glimmer uses 1,2.
export SAM3_GPU=3 MOLMO_GPU=3 EGL_DEVICE=3
bash services/start_perception.sh
```

Keep GPU 0's existing job untouched. Start only one evaluation episode at a time.
Do not stop other users' services when changing model endpoints.

## Check, run, and inspect

**Run robot episodes only after the live Glimmer Python check passes and the
perception services are ready.**
Keep the exported variables in the same terminal, or re-export them in a tmux
session. Set `POLICY_URL=http://127.0.0.1:8600/v1` and `POLICY_MODEL=glimmer`. The readiness check validates interpreter versions, imports, `bwrap`
presence, policy/Molmo model endpoints and SAM route availability. It does not
prove segmentation inference, bubblewrap permissions or EGL work; the first
episode is the integration test.

```bash
cd /data3/may/muse-glimmer-ppo
export POLICY_URL=http://127.0.0.1:8600/v1 POLICY_MODEL=glimmer
python3 scripts/captain_baseline.py check

# Optional: prints the six upstream commands; launches no episodes.
python3 -u scripts/captain_baseline.py run \
  --model-source meta-models/Muse-Glimmer-30B \
  --out runs/glimmer_captain_zeroshot_task0_dry --dry-run

# Run in tmux so a lost SSH connection does not interrupt evaluation.
python3 -u scripts/captain_baseline.py run \
  --model-source meta-models/Muse-Glimmer-30B \
  --out runs/glimmer_captain_zeroshot_task0 --parallel 1
```

This calls their `eval/run_jobs.py`, which calls their unchanged
`eval/run_episode.sh`. Settings include temperature 1, top-p 1, thinking enabled,
60 requests, 150 physical actions, 1080-second wall budget, 64k context, 12 images,
and the authors' prompts, Python schema, compaction and completion gate.

The official runner writes per-episode logs instead of streaming every request
to this terminal. From another terminal:

```bash
cd /data3/may/muse-glimmer-ppo
python3 scripts/captain_baseline.py summary --out runs/glimmer_captain_zeroshot_task0
tail -f runs/glimmer_captain_zeroshot_task0/logs/libero_30.log
```

The task 0 / seed 11 official row has ID 30 at the pinned revision. The summary
reports completed episodes, host success, strict success (host success plus
`finish()`), termination reasons, and missing/infrastructure cases. Infrastructure
errors are not counted as robot-policy failures. Our existing PPO viewer does
not yet read this upstream output format.

For the old-harness comparison, evaluate the **same base Glimmer weights** on
these same initial states using `scripts/eval_task.py`. Our previous SFT
checkpoint may be evaluated as a separate condition, but it was trained on the
move-only interface, not on this Python interface. Do not mix those conditions.

For the full official evaluation, prepare a **separate plan** and pass it to all
subsequent `check`, `run`, and `summary` commands:

```bash
python3 scripts/captain_baseline.py prepare --full --plan data/captain/official240
```

## Training and adding our changes

Frozen Glimmer evaluation tests the interface change. The next training step
uses CaPtain's in-harness trajectory format for **Glimmer SFT**, then adapts the
physical-critic RL stage to Glimmer. The authors' entry points and configs under
`sft/` and `rl/` are references, not drop-in Glimmer trainers: model loading,
chat templates, vision freezing, supervised masks, sampling and reference-policy
scoring must be adapted and checked.

The released SFT run used 16 H100-80GB GPUs. RL provides 16- and 8-GPU layouts;
there is no validated four-Ada Glimmer recipe. A smaller hardware configuration,
QLoRA, or context reduction is an explicit adaptation, not an exact reproduction.
Our move-only Astra examples are not in-harness Python demonstrations.

After establishing Glimmer's harness baseline, add one change per comparison:
in-harness SFT, physical-critic RL, then our alternative rewards or assistance.
Keep task states and settings fixed within each comparison. Report improvement
over **Glimmer's own zero-shot and SFT baselines**. The paper's reported Qwen
numbers are separate reference results, not Glimmer's baseline.
