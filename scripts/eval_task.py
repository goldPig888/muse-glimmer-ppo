"""Evaluate Muse (base, or with a trained adapter) on held-out initial states.

    python scripts/eval_task.py --output runs/eval_base --task 0 --inits 0-9
    python scripts/eval_task.py --output runs/eval_u50 --task 0 --inits 0-9 \
        --adapter runs/task0/checkpoints/update_0050

Same protocol as training (paper prompt, tools, 80 calls, 900 steps); the
adapter is loaded on top of the frozen base. Writes results.jsonl and prints
the success rate and partial-progress rates.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import torch

from glimmer_ppo.paper_env import PaperEnv, PaperEnvConfig
from glimmer_ppo.policy import Policy, PolicyConfig
from glimmer_ppo.sim.embodiment import Start
from glimmer_ppo.starts import parse_ids
from glimmer_ppo.assistance import AssistedStartSampler, AssistanceConfig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--inits', default='0-9')
    p.add_argument('--samples', type=int, default=1, help='episodes per initial state (for pass@k)')
    p.add_argument('--adapter', type=Path)
    p.add_argument('--model', default=PolicyConfig.model)
    p.add_argument('--load-in-4bit', action='store_true')
    p.add_argument('--max-new-tokens', type=int, default=2048)
    p.add_argument('--max-prompt-tokens', type=int)
    p.add_argument('--reasoning', default='high')
    p.add_argument('--object-references', type=Path,
                   help='evaluate full demo initial states instead of benchmark --inits')
    p.add_argument('--assist-gain', type=float, default=0., help='default zero: assistance-free evaluation')
    p.add_argument('--seed', type=int, default=123)
    args = p.parse_args()
    if args.samples < 1:
        p.error('--samples must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Use a fresh evaluation output directory')
    if args.assist_gain != 0 and not args.object_references:
        p.error('--assist-gain requires --object-references')
    starts = [Start('libero_10', args.task, init, label=f'init{init:02d}') for init in parse_ids(args.inits, 50)]
    if args.object_references:
        sampler = AssistedStartSampler(args.object_references, args.task, AssistanceConfig(gain=args.assist_gain))
        # Evaluation never advances an assistance curriculum.
        starts = [sampler.start_at(i) for i in range(len(sampler.references))]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'evaluation_config.json').write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))
    policy = Policy(PolicyConfig(model=args.model, load_in_4bit=args.load_in_4bit,
                                 max_new_tokens=args.max_new_tokens, max_prompt_tokens=args.max_prompt_tokens,
                                 reasoning_strength=args.reasoning or None))
    if args.adapter:
        policy.model.load_adapter(str(args.adapter), adapter_name='trained')
        policy.model.set_adapter('trained')
    policy.eval()
    env = PaperEnv(args.output / 'episodes', PaperEnvConfig())
    rows = []
    for si, start in enumerate(starts):
        for k in range(args.samples):
            torch.manual_seed(args.seed + si * args.samples + k)
            messages = env.reset(replace(start, label=f'{start.label}_s{k}'))
            done, info = False, {}
            with torch.no_grad():
                while not done:
                    text, truncated, _ = policy.act(messages, env.tools)
                    _, done, info = env.step(text, truncated)
                    messages = env.messages
            row = dict(init=start.label if args.object_references else start.init_state,
                       sample=k, success=info['success'], outcome=info['outcome'],
                       decisions=info['decisions'], steps=info['steps'], goal_fraction=info['goal_fraction'],
                       grasped=bool(info['grasped']), lifted=bool(info['lifted']))
            if 'voc' in info:
                row['voc'] = info['voc']
                if args.assist_gain == 0 and (info['voc']['max_force_N'] != 0 or info['voc']['max_torque_Nm'] != 0):
                    raise RuntimeError('Assistance-free evaluation applied helper forces')
            rows.append(row)
            with (args.output / 'results.jsonl').open('a') as f:
                f.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
    env.close()
    n = len(rows)
    by_init = {}
    for r in rows:
        by_init.setdefault(r['init'], []).append(r['success'])
    summary = dict(
        episodes=n, success_rate=sum(r['success'] for r in rows) / n,
        pass_at_k=sum(any(v) for v in by_init.values()) / len(by_init),
        mean_goal_fraction=sum(r['goal_fraction'] for r in rows) / n,
        grasp_rate=sum(r['grasped'] for r in rows) / n,
        start_kind='full_demo_initial_state' if args.object_references else 'benchmark_initial_state',
        assistance_gain=args.assist_gain if args.object_references else 0.)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
