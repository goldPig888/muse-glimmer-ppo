# Train Glimmer with CaPtain's physical critic

This is an RL training entry point, not a frozen-policy check. The actor is
Muse Glimmer 30B. It uses the pinned CaPtain Python harness, the authors' physical
feature extractor and critic architecture, and their same-time GAE computation.
No Qwen language-policy checkpoint is downloaded or used.

## Start on cs94085

Stop the old **standalone** `serve_glimmer_python.py` process with Ctrl+C in its
terminal first. Training owns its own model and HTTP endpoint. Leaving the old
model resident would consume the same GPUs.

```bash
cd /data3/may/muse-glimmer-ppo
git pull --ff-only
tmux new -s glimmer-rl
```

Inside tmux:

```bash
cd /data3/may/muse-glimmer-ppo
export PPO_PY=/data3/may/miniconda3/envs/llm-agent-memory/bin/python
bash scripts/start_captain_rl.sh runs/task0_captain_rl --rounds 5
```

The first launch installs the missing simulator, persistent Python interpreter,
SAM service, and Molmo/vLLM service into **separate** environments. It does not
replace the existing Glimmer environment. Subsequent launches reuse
`data/captain/training.env`. Installation and model downloads can take time;
`[setup]` output is dependency setup, not RL progress.

Requirements for that first install:

- Conda (the machine already has it), Git and curl.
- `bubblewrap`; if missing, install with `sudo apt-get install bubblewrap`.
- Hugging Face access to `facebook/sam3`. Accept the model's access terms and
  authenticate locally if the download is gated. An exported `HF_TOKEN` is used
  without being written to the setup configuration. Do not paste tokens into chat.

The default placement is Glimmer training on physical GPUs **1,2**, and SAM,
Molmo and EGL rendering on **3**. GPU 0's existing job is untouched. Set
`LEARNER_GPUS`, `SAM3_GPU`, `MOLMO_GPU`, and `EGL_DEVICE` before launching to
change placement. The Ada GPU memory fit and real robot rollouts still need to
be established on cs94085; this code has not been GPU-qualified on that machine.

No separate smoke/evaluation experiment runs. The upstream-required sandbox
validation is performed automatically at training startup and saved inside the
training run. If environment or service startup fails, training stops with an
error instead of turning the infrastructure failure into a policy reward.

## What each round does

1. Select one of the authors' 100 fresh task-0 initial states, seeds >=1000.
   These are full-task starts, not demo keyframes. Official states 11–16 remain
   outside this new training bank. The bank's SHA-256 is checked.
2. Collect **four independent Glimmer episodes from the same initial state**,
   sequentially. Every episode uses the persistent Python tool, camera images,
   perception/geometry API, guarded robot motions, receipts, and completion gate.
3. Score physical request boundaries with the released five-member critic,
   using 36 physical features, fixed request time and the task one-hot. The
   critic never reads the actor's hidden state and its inputs are hidden from
   the actor's sandbox.
4. Use same-time value differences, terminal returns and costs with GAE lambda
   0.9. Combine group-relative episode return with centered within-episode GAE.
   Text-only terminal replies have no physical mark: they receive group outcome
   credit only. This also keeps no-tool policy failures in the group statistics.
5. Uniformly sample at most eight requests per trajectory, applying inverse
   inclusion weights. Compute behavior and reference log-probabilities for the
   exact sampled tokens and stored multimodal inputs before any optimizer step.
6. Accumulate one actor update using the SAO-style detached ratio mask (.2, 4),
   trajectory-mean weighting and k3 KL (beta .002). Pure code failures receive
   no positive task credit and get the local code-error penalty. No-signal
   batches do not take an actor optimizer step; this is recorded explicitly.
7. Fit the separate physical critic with two AdamW BCE steps on Glimmer's new
   episode success labels. Credit above always uses the critic **before** this
   update. Save a resumable checkpoint for the completed round.

The terminal returns are +1 for success with finish, +.25 for success without
finish, -.3 for false finish, -.1 for exhausted budgets, otherwise 0. The authors'
action-cost references, failed-action costs and bounded post-success costs are
used. The logged `mean_return` includes those physical/request costs; it does
not use our old potential-based PPO reward. Task 0 is not one of the authors'
staged-reward tasks. No object assistance is applied.

## Differences from the paper's full run

- Base **Glimmer**, as requested, rather than an in-harness SFT Qwen actor.
- NF4 QLoRA rather than full-parameter fine-tuning.
- One same-start group of four episodes per update, sequential on the available
  machine, instead of the paper's large streaming/distributed batches.
- Full-distribution Hugging Face sampling, so no top-k score-centering correction
  or separate rollout-engine quantization is needed. Collection and scoring use
  the same policy precision and weight version.
- KL is to frozen base Glimmer (adapter disabled), not to the paper's SFT model.
- The released critic was calibrated on earlier Qwen-lineage rollouts. Its exact
  bytes and feature spec are hash-checked, but its initial calibration for Glimmer
  is unknown; it is co-trained on new Glimmer outcomes.
- No critic-based early termination. The ordinary request/action/wall budgets
  still apply. Defaults: 60 requests, 120 physical actions, 1080 seconds.

These are recorded in `config.json`. This is a runnable adaptation of the method,
not a claim to reproduce the paper's success rates or hardware throughput.

## Progress and viewer

The terminal prints rollout results, token tqdm bars, RL backward requests and
checkpoints. Training records:

- `progress.json`: round/rollout, generation tokens, scoring/backward stage, errors.
- `metrics.jsonl`: round success, return, actor objective, gradient norm, KL,
  whether an actor step happened, and physical critic BCE.
- `episodes.jsonl`: every valid completed training rollout, success and verdict.
- `rollouts/round_*/sample_*`: the upstream episode, Python/tool receipts, images,
  goal audits and hidden physical marks; the actor cannot access these host files.
- `requests.jsonl`: raw generated replies and parsed Python calls.
- `checkpoints/round_*`: actor adapter, both optimizers, physical critic, RNG state
  and round counter. No model weights are overwritten in the original cache.

```bash
cd /data3/may/muse-glimmer-ppo
cat runs/task0_captain_rl/progress.json
tail -n 3 runs/task0_captain_rl/metrics.jsonl
tail -n 3 runs/task0_captain_rl/episodes.jsonl
```

Restart the viewer once to load the new run type, then select `task0_captain_rl`.
Its training-only dropdown includes this run. Generation tokens, backward
progress and round curves are displayed. This is training success, not a held-out
accuracy measurement. The upstream camera stream is not reconstructed in this
view; saved episode images remain under the rollout directories.

Detach with Ctrl+B, then D. Reattach with `tmux attach -t glimmer-rl`.
Ctrl+C interrupts the active round; completed checkpoints remain. Partially
collected rounds are not resumed. To continue after a saved round into a fresh
run folder, keeping the same training options:

```bash
export PPO_PY=/data3/may/miniconda3/envs/llm-agent-memory/bin/python
bash scripts/start_captain_rl.sh runs/task0_captain_rl_resume \
  --rounds 10 --resume runs/task0_captain_rl/checkpoints/round_0005
```

`--rounds` is the total target, so that command runs rounds 6–10. Resume validates
critical configuration and restores actor/critic optimizer and RNG states; it
is not a promise of bitwise reproducibility across different GPU hardware.

## Validation scope

CPU integration exercises the real released critic and feature extractor, a
synthetic grouped actor update, critic BCE updates, checkpoint saving, and
continuing from restored actor/critic optimizers. Protocol tests cover native
image/tool rendering, request identity, incomplete calls and inference-tensor
capture. This validation does not establish memory fit, service installation,
real Glimmer gradients or task success on the remote GPUs. The first remote
training run is the required end-to-end hardware validation.
