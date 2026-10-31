"""Validate demo starts in LIBERO before loading Muse; save only usable states."""
import argparse
import json
from pathlib import Path

import numpy as np

from glimmer_ppo.paper_env import PaperEnv, PaperEnvConfig
from glimmer_ppo.starts import StartSampler


def rejection_reason(progress, success, min_fraction, require_grasp):
    fraction = float(progress.get('fraction', 0))
    if success or fraction >= 1:
        return 'already successful; no completion left to learn'
    if fraction < min_fraction:
        return 'not enough goal conditions satisfied for this stage'
    if require_grasp and not progress.get('grasped_objects_of_interest'):
        return 'target object not held after restoring and settling'
    return None


def release_in_place(env, max_calls=1, retreat_m=0.0):
    """Physics baseline only; selected training states remain the original pre-release states."""
    total_reward = 0.0
    for call in range(max_calls):
        targets = {'gripper': 0.0}
        if call == 2 and retreat_m:
            targets['z'] = float(env.embodiment.last['eef_state'][2]) + retreat_m
        reply = json.dumps(dict(name='move_to', arguments=dict(
            targets=targets, note='Validation: release, optionally clear the hand upward, and hold open.')))
        reward, done, info = env.step(reply)
        total_reward += reward
        if done or info['success']:
            break
    return dict(release_success=bool(info['success']), release_reward=total_reward,
                release_status=info['status'], release_goal_fraction=info['goal_fraction'],
                release_calls=call + 1, release_retreat_m=retreat_m)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--demo-states', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='validated .npz; training must use this file')
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--kind', choices=['pre_grasp', 'pre_release'], default='pre_release')
    p.add_argument('--max-starts', type=int, default=5)
    p.add_argument('--min-goal-fraction', type=float, default=.5)
    p.add_argument('--require-grasp', action='store_true')
    p.add_argument('--test-release', action='store_true',
                   help='also require a scripted release test to complete the task')
    p.add_argument('--release-calls', type=int, default=1,
                   help='maximum calls in the release test; repeated open commands allow the fingers to finish opening')
    p.add_argument('--release-retreat', type=float, default=0.0,
                   help='meters to retreat upward on test call 3, after two opening calls; default no arm motion')
    args = p.parse_args()
    if args.max_starts < 1 or not 0 <= args.min_goal_fraction < 1:
        p.error('require --max-starts >= 1 and 0 <= --min-goal-fraction < 1')
    if (args.release_calls < 1 or not 0 <= args.release_retreat <= .1
            or (args.release_retreat and args.release_calls < 3)):
        p.error('require --release-calls >= 1; retreat must be 0..0.1 m and needs at least 3 calls')
    if args.output.resolve() == args.demo_states.resolve() or args.output.exists():
        p.error('choose a fresh output .npz; do not overwrite source states')
    sampler = StartSampler('libero_10', [args.task], [10], args.demo_states, 1,
                           kinds=(args.kind,), allow_test_suite=True)
    with np.load(args.demo_states, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    indices = [i for i, (task, kind) in enumerate(zip(arrays['task_id'], arrays['kind']))
               if int(task) == args.task and str(kind) == args.kind]
    accepted, report = [], []
    env = PaperEnv(args.output.with_suffix('') / 'episodes',
                   PaperEnvConfig(max_calls=max(2, args.release_calls + 1)))
    try:
        for index, start in list(zip(indices, sampler.demo))[:args.max_starts]:
            try:
                env.reset(start)
                progress = env.progress
                success = bool((env.embodiment.last or {}).get('success', False))
                reason = rejection_reason(progress, success, args.min_goal_fraction, args.require_grasp)
                row = dict(index=index, label=start.label, accepted=reason is None, reason=reason,
                           fraction=progress.get('fraction'),
                           held=progress.get('grasped_objects_of_interest', []))
                if row['accepted'] and args.test_release:
                    row.update(release_in_place(env, args.release_calls, args.release_retreat))
                    if not row['release_success']:
                        row.update(accepted=False, reason='scripted release test did not complete the task')
            except Exception as exc:
                row = dict(index=index, label=start.label, accepted=False,
                           reason=f'{type(exc).__name__}: {exc}')
            report.append(row)
            print(json.dumps(row), flush=True)
            if row['accepted']:
                accepted.append(index)
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.report.json').write_text(json.dumps(report, indent=2))
    if not accepted:
        raise SystemExit('No usable starts; inspect the validation report before training')
    n = len(arrays['task_id'])
    selected = {k: v[accepted] if v.ndim > 0 and v.shape[0] == n else v for k, v in arrays.items()}
    np.savez_compressed(args.output, **selected)
    print(f'Saved {len(accepted)} validated starts to {args.output}', flush=True)


if __name__ == '__main__':
    main()
