# Experiment viewer and history

The viewer reads existing files and observes owned processes. It does not load
model weights, write to experiment folders, train, stop, or restart Glimmer.

## What has been tried

These notes summarize the folders and results previously pasted from cs94085.
They are historical context, not a fresh evaluation. The selected run's actual
configuration, metrics and result files are shown separately and take precedence.

| Run folder | Experiment / previous observation |
|---|---|
| `check_paper_env` | Environment/tool smoke test. Exercises valid and invalid moves; this is not policy training. |
| `task0_ada` | Original full-task Glimmer PPO attempt on the Ada machine. Earlier check: two saved updates, then the process stopped. |
| `task0_ada_resume` | Continuation of the original PPO attempt from update 2. Started from normal task initial states. |
| `task0_curriculum_release15` | Reverse-curriculum attempt using demo states before the final release. The folder name alone does not establish how much training completed; inspect its configuration and records. |
| `task0_curriculum_release1` | Very near final-release curriculum, with the first object already placed and the second held. This tests completion of the ending, not the full task from the beginning. |
| `task0_curriculum_release2` | Final-release curriculum starting two recorded demo frames before release. Earlier displayed run: 7 successes out of 8 completed episodes. |
| `compare_lead2_11_12` | Frozen comparison of release-2 checkpoints 11 and 12. Earlier report: each succeeded in 8/10 trials, with different failing starts; no clear winner from that small comparison. |
| `task0_astra_sft_trial` | Small Astra imitation-learning trial. Earlier report: 64 training examples, one epoch; validation token NLL 0.5574 to 0.5349. This did not establish robot success. |
| `task0_astra_sft_full` | Full prepared Astra task-0 imitation dataset. Earlier report: 611 training examples, one epoch; validation NLL 0.5709 to 0.4319. |
| `task0_astra_sft_ppo_release2` | Despite its folder name, this PPO attempt was launched from normal initial states after the trial SFT checkpoint, without the release-state arguments. Check the saved configuration before interpreting it as a curriculum run. |
| `task0_sft_full_eval_8_9` | Frozen full-task evaluation of the full SFT checkpoint on initial states 8 and 9, two samples each. Earlier report: 0/4 successes, final goal fraction 0. Final grasp flags do not prove the robot never grasped during a rollout. |
| `task0_object_force_check` | Object-assistance physics check with an idle robot. Earlier check: helper gain 1 completed the task; gain 0 did not. This verifies helper forces, not Glimmer learning. |
| `glimmer_python_server` | First Glimmer Python-tool server launch folder. Server startup or a saved configuration is not a completed robot evaluation. |
| `glimmer_python_server_20261103_094423` | Dated frozen Glimmer Python-tool server launch. Activity and replies are read from this folder and the matching process. Robot episodes require a separate CaPtain evaluation run. |

## List saved experiment data on cs94085

```bash
cd /data3/may/muse-glimmer-ppo
python3 scripts/serve_training_viewer.py --runs-root runs --list

# Start-state files and prepared teacher datasets (filenames only):
find data -maxdepth 2 -type f \( -name '*.npz' -o -name 'manifest.json' -o -name 'examples.jsonl' \) -print
```

`--list` reports the real folder paths, detected kind, matching process activity,
and metadata-file modification times. It reads no model weights. Blank or
unrecognized top-level attempt folders remain selectable and say that they have
no supported records. New folders are discovered automatically every five seconds.

## Start the all-experiments viewer

Use a new port so the old viewer and current experiment can keep running:

```bash
cd /data3/may/muse-glimmer-ppo
git pull --ff-only
nohup python3 scripts/serve_training_viewer.py \
  --runs-root /data3/may/muse-glimmer-ppo/runs --port 8770 \
  > runs/viewer_all.log 2>&1 &
```

The page is `http://127.0.0.1:8770` **on cs94085**. Select an experiment by folder
name at the top. The URL retains the selection, so it can be bookmarked.
`--run runs/NAME` optionally chooses the initial experiment. The old single-run
command remains supported and discovers sibling experiments.

From another PC, either forward this port over SSH, or start a separate
Cloudflare tunnel. If `/tmp/cloudflared` is still the binary used before:

```bash
nohup /tmp/cloudflared tunnel --url http://127.0.0.1:8770 --protocol http2 \
  > runs/viewer_all_cloudflare.log 2>&1 &
```

After it starts, print its new link:

```bash
grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' runs/viewer_all_cloudflare.log
```

The previous tunnel URL still points at the old port; use the new link for this
viewer. No training/model process needs to be stopped. To check startup:

```bash
tail -n 20 runs/viewer_all.log
curl -fsS http://127.0.0.1:8770/api/runs
```

## What each selection shows

- **PPO:** existing reward/action curves, optimizer progress, episode picker,
  camera frames and per-decision inspection. Model checkpoints stay listed in
  their experiment, rather than appearing as separate runs.
- **SFT:** training and validation token NLL curves, epoch progress, example
  counts, metrics, config and saved epoch checkpoints. NLL is not robot accuracy.
- **Evaluation and checkpoint comparisons:** completed results and failures,
  cumulative success (not a learning curve), saved summaries including each
  checkpoint/start, and episode frames/actions where they were saved.
- **Object-force checks:** gain-on and gain-off physics results. Idle/helper
  success is labeled separately from policy performance.
- **Glimmer Python server:** process status, reply/call/error counts, generated
  tokens, request time, raw replies and parsed Python calls. No rewards are
  fabricated. An active server with no requests may still be loading or waiting.
- **Python smoke checks / critic diagnostics:** saved check/report records.
- **CaPtain robot evaluation:** saved host/strict success, termination reasons
  and infrastructure-error counts. This view does not yet reconstruct the
  upstream driver's image stream.

Only existing data can be displayed. Missing terminal log files or camera frames
are marked missing, never borrowed from another experiment. Historical PPO run
time stops at the last saved activity. GPU usage is **the machine right now**,
not historical consumption by the selected experiment. Request/episode records
in non-PPO views show the latest 100 entries, with the full total displayed.

The running Python server already saves complete replies, so they appear without
a restart. Its live tqdm token bar is terminal-only and is not retroactively
available as a file to this viewer. Model loading or a Python server's existence
must not be interpreted as robot evaluation progress.
