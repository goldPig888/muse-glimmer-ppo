# muse-glimmer-ppo

PPO fine-tuning of Muse Glimmer 30B as a robot policy: Muse writes harness
programs (hover, grasp, place, ...) for a simulated Panda, LIBERO scores the
result, and PPO updates LoRA adapters on Muse's language model plus a value
head. Training uses LIBERO-90; LIBERO-10 stays held out for testing against
the zero-shot results in [llm-agent-memory](https://github.com/goldPig888/llm-agent-memory).

Plan, parameters and the PPO walkthrough: the "Glimmer RL on LIBERO-90: Plan" doc.

## How it fits together

```text
StartSampler ──> HarnessEnv.reset(start) ──> chat messages (system + task + 2 camera images)
                                               │
                 Policy.act(messages) <────────┘   Muse + LoRA samples one reply
                         │
HarnessEnv.step(reply) ──┘  parse curl -> Session.execute -> worker.py (LIBERO, MuJoCo)
        │                   reward = success + progress + first grasp - rejections - repeats
        v
rollout buffer ──> GAE per decision ──> clipped PPO epochs ──> LoRA + value head
```

| File | What it does |
|---|---|
| `src/glimmer_ppo/sim/worker.py` | LIBERO subprocess. Any suite; starts from an init state or a saved sim state; every reply carries BDDL goal progress and whether an object of interest is grasped (reward only, never shown to the model) |
| `src/glimmer_ppo/sim/embodiment.py` | Talks to the worker; drop-in robot for the harness `Session` |
| `src/glimmer_ppo/env.py` | The harness as an RL environment: one step = one model reply |
| `src/glimmer_ppo/prompt.py` | Same system prompt as the wrist-v3 zero-shot runs (copied from `run_harness.py` at 48989e2) |
| `src/glimmer_ppo/reward.py` | Per-decision reward terms |
| `src/glimmer_ppo/policy.py` | Muse with LoRA (text decoder only), value head, reference = adapter off |
| `src/glimmer_ppo/ppo.py` | GAE, clipped policy loss, value loss, KL, explained variance |
| `src/glimmer_ppo/starts.py` | Samples LIBERO-90 init states and mid-demo starts; refuses LIBERO-10 |
| `scripts/train_ppo.py` | The training loop |
| `scripts/check_env.py` | Environment and reward check, no model |
| `scripts/print_modules.py` | Lists Muse's linear layers and the LoRA targets, no weights |
| `scripts/extract_start_states.py` | Pre-grasp and pre-release states from LIBERO-90 demos |

## Setup

Clone next to `llm-agent-memory` (its harness, parser, checker and LIBERO
environment are reused), then install both into the same environment:

```bash
conda activate llm-agent-memory           # the env from llm-agent-memory/scripts/00_setup_env.sh
pip install -e ../llm-agent-memory
pip install -e ".[dev,demos]"             # add ,qlora for 4-bit
huggingface-cli download meta-models/Muse-Glimmer-30B   # bf16 weights; the GGUF cannot be trained
```

## Order of work

1. **Tests** (no GPU, no simulator): `python -m pytest`
2. **Environment**: `python scripts/check_env.py --list`, then
   `python scripts/check_env.py --task 0 --init 0`. Expect three rewards, a
   video in `runs/check_env`, and the task's goal predicates printed.
3. **LoRA targets**: `python scripts/print_modules.py`. The default targets
   are `q/k/v/o_proj` and `gate/up/down_proj` outside the vision tower and
   projector. If Muse names its layers differently, change `lora_targets` in
   `policy.py`.
4. **Mid-demo starts**: download the LIBERO-90 demos, then
   `python scripts/extract_start_states.py --demos <folder> --output data/libero90_starts.npz`
5. **Stage 2, one-decision PPO from pre-grasp states**:
   ```bash
   python scripts/train_ppo.py --output runs/stage2 --max-decisions 1 \
       --demo-states data/libero90_starts.npz --demo-fraction 1.0 \
       --decisions-per-update 64 --updates 50
   ```
6. **Stage 3**: `--max-decisions 10 --demo-fraction 0.7`, then **stage 4**:
   full episodes, `--max-decisions 80 --demo-fraction 0`.

Read `runs/<name>/metrics.jsonl` after each update: `mean_reward`,
`success_rate`, `approx_kl` (should stay small), `clipfrac` (roughly 0.1 to
0.3 is healthy), `explained_variance` (should rise toward 1 as the critic
learns), `truncated` and `rejected` (collapse shows up here first).

## Design choices

- **Our own PPO loop, same math as TRL's PPO.** TRL's `PPOTrainer` is
  experimental and built for single-turn text with a reward model. Here one
  step is one harness decision with images and a simulator reward, so the loop
  is written out (`ppo.py` + `train_ppo.py`, about 300 lines) with TRL's
  conventions: KL to the reference subtracted from the reward, clipped value
  loss, advantage whitening.
- **Credit per decision.** GAE runs over decisions; every token of a program
  shares that decision's advantage.
- **Value at the last prompt token**, from the final hidden state, through a
  linear head. No second 30B critic.
- **Log-probs for PPO come from the training model**, never from llama.cpp or
  vLLM, so the ratio is exact.
- **Critic warm-up**: the first `--critic-warmup-updates` updates train only
  the value head, so early advantages are not pure noise.
- **Privileged reward, unprivileged policy.** Goal predicates and the grasp
  check read simulator state; the model sees only the same images and robot
  state as in the zero-shot runs.

## Status

The PPO math, reward and LoRA target selection are unit-tested (23 tests).
`check_env.py` has been run on LIBERO-90 task 0 (close the top drawer): reset,
goal predicates, an observation turn, a 60-step hover through the checker, and
a malformed reply all produced the expected rewards. The policy and training
loop have not yet been run with Muse: run steps 1 to 3 first and expect to fix
model-loading details (processor arguments, layer names) on the first real run.
