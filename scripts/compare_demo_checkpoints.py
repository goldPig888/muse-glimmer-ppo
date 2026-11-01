"""Compare frozen checkpoints on identical demo starts and sampling seeds.

Loads the base model once. No PPO scores, gradients, or optimizer steps.
Each checkpoint gets every selected start with the same per-trial seeds.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from glimmer_ppo.checkpoint import checkpoint_config
from glimmer_ppo.paper_env import PaperEnv, PaperEnvConfig
from glimmer_ppo.policy import Policy, PolicyConfig
from glimmer_ppo.reward import PotentialConfig
from glimmer_ppo.starts import StartSampler


def summarize(rows):
    def stats(group):
        return dict(episodes=len(group), successes=sum(r['success'] for r in group),
                    success_rate=sum(r['success'] for r in group) / len(group),
                    mean_return=sum(r['return_'] for r in group) / len(group),
                    mean_decisions=sum(r['decisions'] for r in group) / len(group),
                    mean_goal_fraction=sum(r['goal_fraction'] for r in group) / len(group))
    checkpoints = list(dict.fromkeys(r['checkpoint'] for r in rows))
    summary = dict(checkpoints=[])
    for checkpoint in checkpoints:
        group = [r for r in rows if r['checkpoint'] == checkpoint]
        summary['checkpoints'].append(dict(checkpoint=checkpoint, **stats(group),
            per_start=[dict(start=start, **stats([r for r in group if r['start'] == start]))
                       for start in dict.fromkeys(r['start'] for r in group)]))
    if len(checkpoints) == 2:
        a, b = checkpoints
        first = {(r['start'], r['sample']): r for r in rows if r['checkpoint'] == a}
        pairs = [(first[(r['start'], r['sample'])], r) for r in rows
                 if r['checkpoint'] == b and (r['start'], r['sample']) in first]
        summary['paired'] = dict(first=a, second=b, trials=len(pairs),
            first_only_success=sum(x['success'] and not y['success'] for x, y in pairs),
            second_only_success=sum(y['success'] and not x['success'] for x, y in pairs),
            both_success=sum(x['success'] and y['success'] for x, y in pairs),
            both_failure=sum(not x['success'] and not y['success'] for x, y in pairs))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    p.add_argument('--demo-states', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=2, help='repeats per start per checkpoint')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--no-token-progress', action='store_true')
    args = p.parse_args()
    if args.samples < 1 or len(args.checkpoints) < 2:
        p.error('require at least two checkpoints and --samples >= 1')
    if len(set(path.resolve() for path in args.checkpoints)) != len(args.checkpoints):
        p.error('checkpoints must be distinct')
    if args.output.exists() and (not args.output.is_dir()
                                or any(path.name != 'compare.log' for path in args.output.iterdir())):
        p.error('--output must be a fresh folder; existing evaluation results are preserved')
    configs = [checkpoint_config(path) for path in args.checkpoints]
    if any(c.get('args', {}).get('env', 'paper') != 'paper' for c in configs):
        p.error('this comparison supports paper-mode checkpoints only')
    if any(c['policy'] != configs[0]['policy'] or c['env'] != configs[0]['env']
           or c.get('args', {}).get('task', '0') != configs[0].get('args', {}).get('task', '0')
           for c in configs[1:]):
        p.error('checkpoints must have matching model, sampling, task, reward, and episode settings')
    source = configs[0]
    pcfg = PolicyConfig(**source['policy'])
    pcfg.gradient_checkpointing = False
    pcfg.token_progress = not args.no_token_progress
    ecfg = dict(source['env'])
    ecfg['reward'] = PotentialConfig(**ecfg.get('reward', {}))
    env_config = PaperEnvConfig(**ecfg)
    task = int(source.get('args', {}).get('task', 0))
    starts = StartSampler('libero_10', [task], [10], args.demo_states, 1,
                         kinds=('pre_release',), allow_test_suite=True).demo
    labels = [start.label for start in starts]
    if len(set(labels)) != len(labels):
        p.error('demo start labels must be unique')
    args.output.mkdir(parents=True, exist_ok=True)
    plan = dict(mode='frozen_checkpoint_comparison', checkpoints=[str(c.resolve()) for c in args.checkpoints],
                demo_states=str(args.demo_states.resolve()), starts=labels, samples=args.samples,
                seed=args.seed, policy=vars(pcfg), env=source['env'])
    (args.output / 'config.json').write_text(json.dumps(plan, indent=2))
    total = len(starts) * args.samples * len(args.checkpoints)
    tqdm.write(f'[compare] {len(starts)} starts x {args.samples} repeats x '
               f'{len(args.checkpoints)} checkpoints = {total} episodes; learning disabled')
    tqdm.write('[compare] loading the base model once')
    policy = Policy(pcfg)
    rows = []
    def progress(**extra):
        data = dict(completed=len(rows), total=total, learning=False, **extra)
        temp = args.output / 'progress.json.tmp'
        temp.write_text(json.dumps(data, indent=2))
        temp.replace(args.output / 'progress.json')
    progress(phase='evaluating')
    with tqdm(total=total, desc='Frozen evaluation', unit='episode', dynamic_ncols=True) as bar:
        for ci, checkpoint in enumerate(args.checkpoints):
            tqdm.write(f'[compare] loading checkpoint: {checkpoint}')
            policy.load(checkpoint)
            policy.requires_grad_(False)
            policy.eval()
            env = PaperEnv(args.output / f'checkpoint_{ci + 1:02d}_{checkpoint.name}' / 'episodes', env_config)
            try:
                for si, start in enumerate(starts):
                    for sample in range(args.samples):
                        seed = args.seed + si * args.samples + sample
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(seed)
                        progress(phase='evaluating', checkpoint=str(checkpoint.resolve()),
                                 start=start.label, sample=sample, seed=seed)
                        tqdm.write(f'[compare] {checkpoint.name} start={start.label} '
                                   f'sample={sample + 1}/{args.samples} seed={seed}')
                        t0 = time.monotonic()
                        messages = env.reset(replace(start, label=f'{start.label}_sample{sample}'))
                        initial = env.progress
                        if float(initial.get('fraction', 0)) >= 1:
                            raise RuntimeError(f'{start.label} is already successful at reset; revalidate states')
                        if not initial.get('grasped_objects_of_interest'):
                            raise RuntimeError(f'{start.label} has no target held at reset; revalidate states')
                        done, episode_return, decisions, tokens = False, 0., 0, 0
                        while not done:
                            with torch.inference_mode():
                                text, truncated, record = policy.act(messages, env.tools)
                            reward, done, info = env.step(text, truncated)
                            episode_return += reward
                            decisions += 1
                            tokens += len(record['response'])
                            tqdm.write(f'[compare] decision={decisions} tool={info.get("tool")} '
                                       f'status={info["status"]} reward={reward:+.4f} success={info["success"]}')
                            messages = env.messages
                        row = dict(checkpoint=str(checkpoint.resolve()), start=start.label, sample=sample,
                                   seed=seed, success=bool(info['success']), outcome=info.get('outcome'),
                                   decisions=decisions, steps=info['steps'], return_=episode_return,
                                   initial_goal_fraction=initial.get('fraction', 0),
                                   goal_fraction=info['goal_fraction'], response_tokens=tokens,
                                   seconds=round(time.monotonic() - t0, 1))
                        rows.append(row)
                        with (args.output / 'results.jsonl').open('a') as f:
                            f.write(json.dumps(row) + '\n')
                        (args.output / 'summary.json').write_text(json.dumps(summarize(rows), indent=2))
                        tqdm.write(json.dumps(row))
                        bar.update(1)
            except Exception as exc:
                progress(phase='error', error=f'{type(exc).__name__}: {exc}')
                raise
            finally:
                env.close()
    progress(phase='complete')
    tqdm.write(json.dumps(summarize(rows), indent=2))
    tqdm.write(f'[compare] complete; results: {args.output / "summary.json"}')


if __name__ == '__main__':
    main()
