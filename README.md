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

Phi  = 0.2*1[holding soup or sauce] + 0.1*1[lifted >= 3 cm] + 1.0*(goal conditions met / 2)     sub-goals
     - 0.1*min(1, reach/0.3 m) - 0.1*min(1, carry/0.3 m)                                        dense
```

- **reach**: hand to the nearest can not yet in the basket (when not holding one).
- **carry**: held can to the basket's contain region.
- Task 0 has two goal conditions, `In(alphabet_soup, basket)` and
  `In(tomato_sauce, basket)`, so one can in the basket is worth 0.5, more
  than holding and lifting (0.3): releasing into the basket raises Phi.
- Only cans that must be moved count as "holding": grabbing the basket earns nothing.
- Everything is one potential, so over an episode the shaping adds up to
  Phi(end) - Phi(start): hovering, sweeping back and forth, or grab-and-drop earn nothing.
- The model's own `done` is never rewarded.

`--reward sparse | subgoal | both` (default `both`) gives the three runs of the ablation.

## Setup

Clone next to `llm-agent-memory` (the paper's Episode, prompt and toolset
are reused), then install both into its environment:

```bash
conda activate llm-agent-memory
pip install -e ../llm-agent-memory
pip install -e ".[dev,demos]"                  # add ,qlora for 4-bit
huggingface-cli download meta-models/Muse-Glimmer-30B
```

## Running on 2x RTX A6000 (2 x 48 GB)

Stop the llama.cpp server you use for the zero-shot runs first (it holds GPU memory).

**Option A, simplest (start here).** Hugging Face does everything, the model
is spread over both GPUs in 4-bit. Slower generation, exactly on-policy.

```bash
python scripts/train_ppo.py --output /metadisk/may/ppo/task0_both --env paper --task 0 \
    --train-inits 10-49 --reward both --load-in-4bit
```

On 2x 24 GB A5000s, add `--activation-offload` and set
`PYTORCH_ALLOC_CONF=expandable_segments:True`. This stores saved backward
activations in CPU RAM and trades transfer time for GPU memory. Scoring
disables the KV cache and captures only the critic's final hidden token,
rather than retaining every layer's full hidden states. Generation and scoring
use bf16 autocast so fp32 LoRA weights do not produce full-size fp32 adapter
activations. Add `--max-prompt-tokens 12288` to bound history by removing the
oldest complete turns while keeping the task and at least two recent turns.
This changes the context available to the policy, but generation and PPO
scoring use the same context. A short smoke test does not verify memory use
once an episode's conversation grows.

To stress-test a recorded episode before a long run:

```bash
python scripts/check_long_context.py --episode <episode-directory> \
    --output runs/context_check --response-tokens 2048 --activation-offload
```

This tests policy/reference scoring and backward gradients on the recorded
conversation with a synthetic full-length response; it does not update the policy.

### Run on another PC with terminal progress

Use the existing `llm-agent-memory` environment with the simulator and a
PyTorch build compatible with that PC's NVIDIA driver. From this checkout:

```bash
conda activate llm-agent-memory
git pull --ff-only
python -m pip install -e '.[qlora,dev,demos]'
python -c 'import torch; print(torch.__version__, torch.version.cuda); print("CUDA available:", torch.cuda.is_available()); print("GPU count:", torch.cuda.device_count())'
python -m pytest -q
python scripts/check_paper_env.py --task 0 --init 10

mkdir -p runs/task0_both_pc2
set -o pipefail
PYTORCH_ALLOC_CONF=expandable_segments:True python -u scripts/train_ppo.py \
  --output runs/task0_both_pc2 --env paper --task 0 --train-inits 10-49 \
  --reward both --load-in-4bit --activation-offload --max-prompt-tokens 12288 \
  --save-every 1 2>&1 | tee runs/task0_both_pc2/train.log
```

Stop if CUDA is unavailable; fix the PyTorch/driver pairing before training.
The sibling `llm-agent-memory` checkout and its simulator assets are required
on the other PC too; this repository does not contain them or model weights.
Choose a fresh output folder for each run; this script does not resume optimizer state.

Progress bars are enabled by default: PPO updates, collected decisions,
optimizer backward passes, and sampled HF tokens. The token bar may end
before its maximum when Muse emits an end token. Status lines announce
model loading, generation, policy/reference scoring, action results,
rewards, optimization, and checkpoint saves. `python -u` makes redirected
output immediate; `tee` displays and saves both output streams. Use
`--no-token-progress` to hide only the token bar or `--no-progress` to hide
all trainer bars while retaining status lines.

For a short first test, use a different output folder and append
`--updates 1 --decisions-per-update 2 --epochs 1 --minibatch 1 --max-decisions 2`.
This verifies the pipeline, not memory use at maximum episode length.

In a second terminal, activate the same environment and start the viewer:

```bash
python scripts/serve_training_viewer.py --run runs/task0_both_pc2 --port 8766
```

Open http://127.0.0.1:8766 on that PC.

**Option B, faster rollouts.** llama.cpp generates on GPU 0, Hugging Face
trains on GPU 1. After every update the LoRA is converted to GGUF and the
server restarts with it, so rollouts always come from the latest policy.

| GPU | Runs | Approx. memory |
|---|---|---|
| 0 | llama-server: Muse Q8_0 GGUF (29.6 GB) + mmproj + KV cache | ~35 GB of 48 |
| 1 | QLoRA PPO: NF4 weights (~18 GB) + LoRA/optimizer + activations | ~35-45 GB of 48 |

```bash
# once: Q8_0 GGUF for rollouts (closer to the training weights than Q4)
huggingface-cli download unsloth/Muse-Glimmer-30B-GGUF --include "*Q8_0*" --local-dir /metadisk/may/models

# check the two engines agree before training (see the script's docstring)
CUDA_VISIBLE_DEVICES=1 python scripts/check_rollout_server.py --load-in-4bit --server-gpu 0 \
    --llama-server ~/llama.cpp/build/bin/llama-server \
    --gguf /metadisk/may/models/<Q8_0 file>.gguf --mmproj <your mmproj .gguf>

# train
CUDA_VISIBLE_DEVICES=1 python scripts/train_ppo.py --output /metadisk/may/ppo/task0_both \
    --env paper --task 0 --train-inits 10-49 --reward both --load-in-4bit \
    --rollout llamacpp --server-gpu 0 \
    --llama-server ~/llama.cpp/build/bin/llama-server \
    --gguf /metadisk/may/models/<Q8_0 file>.gguf --mmproj <your mmproj .gguf> \
    --llama-convert ~/llama.cpp/convert_lora_to_gguf.py
```

`CUDA_VISIBLE_DEVICES=1` keeps training on GPU 1; the server is started on
`--server-gpu 0` by the script. Use an f16/bf16 mmproj if one is available
(your current one is Q4/Q6). Needs a recent llama.cpp (multimodal `/completion`
with `return_tokens`). Out of memory on GPU 1: lower `--max-new-tokens`
(e.g. 1024) or `--reasoning medium`, since sequence length drives memory.

Approximate in option B: rollouts come from the Q8_0 GGUF while training uses
NF4 4-bit, so samples are very slightly off-policy; PPO's ratio is still
computed consistently by the training model on the exact sampled tokens.

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
   python scripts/train_ppo.py --output runs/task0_both --env paper --task 0 --train-inits 10-49 \
       --reward both --demo-states data/task0_starts.npz --demo-fraction 0.7 --max-decisions 80
   ```
7. Evaluate a checkpoint on states 0-9 with `eval_task.py --adapter ...` and
   compare with step 4 and the paper's zero-shot result.

Watch `runs/task0/metrics.jsonl`: `mean_reward`, `success_rate`, `invalid`
(should fall first), `approx_kl` (small), `clipfrac` (about 0.1 to 0.3),
`explained_variance` (rising toward 1).

## Live training viewer

```bash
python scripts/serve_training_viewer.py --run /metadisk/may/ppo/task0_both --port 8766
```

Open http://127.0.0.1:8766 on the training machine. The read-only viewer
refreshes every two seconds with batch progress, camera frames, episode
history, parsed tool calls and raw reasoning, GPU memory, and the training
log. Learning curves appear after completed PPO updates. Frames refresh
after robot actions, rather than during generation. `train.pid` in the run
directory enables process status; the background launcher writes this file.
The viewer uses only Python's standard library and does not load the model.

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

50 tests pass, including offline training-loop progress and scoring-gradient
checks. Muse loads in 4-bit on 2x A5000 24 GB cards; its native XML tool calls
parse and execute in LIBERO. A two-decision PPO update saved a checkpoint.
With bf16 autocast and CPU activation offloading, a recorded 14,402-token
prompt plus a synthetic 2,048-token response passed policy/reference scoring
and backward, with peak PyTorch allocations of 17.86 / 19.15 GiB. This is a
memory stress test, not a completed training run or a demonstrated task success.
