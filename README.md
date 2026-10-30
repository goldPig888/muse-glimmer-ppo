# muse-glimmer-ppo

PPO fine-tuning of Muse Glimmer 30B as a robot policy, starting with the
first task of the LLM-LIBERO paper: LIBERO-10 task 0, "put both the
alphabet soup and the tomato sauce in the basket".

Muse acts exactly as in the paper: each decision is one tool call
(`move_to` with absolute targets, `done` or `give_up`), with the paper's
system prompt, tool schemas, 80-call and 900-step budget, and success check.
PPO updates LoRA adapters on Muse's language model plus a value head.
Training uses initial states 10-49 of the task; states 0-9, the ones the
paper evaluated, stay held out.

The harness version (skill programs, LIBERO-90) is kept in the repo for later.

Plan, formulas and parameters: the "Glimmer RL on LIBERO-90: Plan" doc.

## How it fits together

```text
StartSampler (task 0, inits 10-49, optional demo keyframes)
   │
PaperEnv.reset ──> llm_agent_memory Episode (paper Algorithm 1) ──> paper_worker.py (LIBERO, MuJoCo)
   │                messages: paper system prompt + goal + state + 2 camera images
   v
Policy.act(messages, tools)      Muse + LoRA samples one reply, parse_tool_call reads the call
   │
PaperEnv.step ──> Episode.call(move_to | done | give_up)  (paper toolset: bounds, interpolation)
   │              reward = success + Phi(s') - Phi(s) - costs     (Phi: hold, lift, goal conditions)
   v
rollout buffer ──> GAE per call ──> clipped PPO epochs ──> LoRA + value head
```

| File | What it does |
|---|---|
| `src/glimmer_ppo/paper_env.py` | The paper protocol as an RL env: one step = one reply = one tool call |
| `src/glimmer_ppo/sim/paper_worker.py` | The paper's LIBERO worker, unchanged except it also reports goal conditions, grasp and lift (reward only) and can start from a demo state |
| `src/glimmer_ppo/sim/paper_embodiment.py` | The paper's `LiberoEmbodiment`, pointed at that worker |
| `src/glimmer_ppo/toolcall.py` | Reads the last move_to / done / give_up call from Muse's raw text |
| `src/glimmer_ppo/reward.py` | `potential_reward` (paper env) and the older harness reward |
| `src/glimmer_ppo/policy.py` | Muse with LoRA (text decoder only), value head, reference = adapter off |
| `src/glimmer_ppo/ppo.py` | GAE, clipped policy loss, value loss, KL, explained variance |
| `src/glimmer_ppo/starts.py` | Start states: initial states and demo keyframes |
| `scripts/train_ppo.py` | Training loop (`--env paper` default, `--env harness` for later) |
| `scripts/eval_task.py` | Success rate on held-out initial states, base model or a checkpoint |
| `scripts/check_paper_env.py` | Env and reward check on task 0, no model |
| `scripts/print_modules.py` | Muse's linear layers and the LoRA targets, no weights |
| `scripts/extract_start_states.py` | Pre-grasp and pre-release states from demos |

## Reward (paper env)

```text
r_t  = 1[success] + Phi(s_t+1) - Phi(s_t) - 0.005 - 0.05*1[invalid call] - 0.1*1[repeated invalid call]
Phi  = 0.2*1[holding soup or sauce] + 0.1*1[lifted >= 3 cm] + 1.0*(goal conditions met / 2)
```

Task 0 has two goal conditions, `In(alphabet_soup, basket)` and
`In(tomato_sauce, basket)`, so one can in the basket is worth 0.5. That is
more than holding and lifting (0.3), so releasing into the basket raises Phi.
The model's own `done` is never rewarded. `--no-shaping` drops Phi for the
sparse-success ablation.

## Setup

Clone next to `llm-agent-memory` (the paper's Episode, prompt and toolset
are reused), then install both into its environment:

```bash
conda activate llm-agent-memory
pip install -e ../llm-agent-memory
pip install -e ".[dev,demos]"                  # add ,qlora for 4-bit
huggingface-cli download meta-models/Muse-Glimmer-30B
```

## Order of work for task 0

1. `python -m pytest` (no GPU, no simulator).
2. `python scripts/check_paper_env.py --task 0 --init 0`: a 5 cm move,
   an out-of-bounds move, the same move again and a reply with no call;
   expect the rewards and tool results printed.
3. `python scripts/print_modules.py`: confirm the LoRA targets.
4. Baseline on the held-out states with this code path:
   `python scripts/eval_task.py --output runs/eval_base --task 0 --inits 0-9`.
   Open one episode's `ppo_turns.jsonl` and check `parsed` is filled: if
   Muse's tool-call text has a format `toolcall.py` does not read, add it there.
5. Optional demo starts:
   `python scripts/extract_start_states.py --suite libero_10 --allow-test-suite --tasks 0 --demos <libero_10 folder> --output data/task0_starts.npz`
6. Train:
   ```bash
   python scripts/train_ppo.py --output runs/task0 --env paper --task 0 --train-inits 10-49 \
       --demo-states data/task0_starts.npz --demo-fraction 0.7 --max-decisions 80
   ```
7. Evaluate a checkpoint on states 0-9 with `eval_task.py --adapter ...` and
   compare with step 4 and the paper's zero-shot result.

Watch `runs/task0/metrics.jsonl`: `mean_reward`, `success_rate`, `invalid`
(should fall first), `approx_kl` (small), `clipfrac` (about 0.1 to 0.3),
`explained_variance` (rising toward 1).

## Design choices

- **Paper code for everything the model sees and does.** Episode, prompt,
  tool schemas, bounds and interpolation come from `llm-agent-memory` and the
  pinned LLM_LIBERO and inspect-robots commits; only the reward is added.
- **Our own PPO loop, same math as TRL's PPO.** TRL's `PPOTrainer` is
  experimental and single-turn with a reward model.
- **Credit per tool call;** every token of a call shares its advantage.
- **Value at the last prompt token** through a linear head; no second model.
- **Log-probs for PPO come from the training model** (logits only for the
  response tokens, since the vocabulary is 202k).
- **Held-out initial states:** this is a single-task run, so the test set is
  states 0-9 of the same task, not a different suite. LIBERO-10's demos for
  task 0 may come from scenes close to those states; report that if demo starts are used.

## Status

Tests pass (PPO math, both rewards, tool-call parsing, LoRA target
selection). `check_env.py` (harness) ran on LIBERO-90 task 0 and
`check_paper_env.py` ran on LIBERO-10 task 0 in the simulator. Not yet run
with Muse: the model loading, its tool-call text format and the training loop
need the first real run on a GPU.
