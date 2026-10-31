"""Save mid-task start states from LIBERO human demonstrations.

    python scripts/extract_start_states.py --demos /path/to/libero_90 --output data/libero90_starts.npz

For each demo, finds where the gripper command switches to close (a grasp)
and back to open (a release), and saves the simulator state ``--lead`` steps
before each event: ``pre_grasp`` (hand near the object, still open) and
``pre_release`` (object held, near the destination). Training episodes that
start there are a few decisions long, so PPO gets clear credit.

The demo files are LIBERO's ``<task name>_demo.hdf5`` (data/demo_k/states,
data/demo_k/actions). Download them with LIBERO's download script.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np


def events(gripper: np.ndarray):
    """Indices where the command crosses 0: (-) to (+) is close, (+) to (-) is open."""
    closed = gripper > 0
    change = np.flatnonzero(closed[1:] != closed[:-1]) + 1
    return [(int(i), 'grasp' if closed[i] else 'release') for i in change]


def start_events(gripper, lead, final_release_only=False):
    if lead < 0:
        raise ValueError('--lead must be nonnegative')
    selected = events(gripper)
    if final_release_only:
        selected = [event for event in selected if event[1] == 'release'][-1:]
    # states[t] precedes actions[t]; use the previous command while settling.
    return [(max(0, index - lead), kind,
             1.0 if gripper[max(0, index - lead - 1)] > 0 else -1.0)
            for index, kind in selected]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--demos', type=Path, required=True, help='folder containing the suite\'s *_demo.hdf5 files')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--suite', default='libero_90')
    p.add_argument('--lead', type=int, default=15, help='steps before the event (20 Hz)')
    p.add_argument('--max-demos', type=int, default=50)
    p.add_argument('--demo-ids', help='only these demonstration ids, e.g. 10-49; not initial-state ids')
    p.add_argument('--final-release-only', action='store_true',
                   help='save only the final release event of each demo for a reverse curriculum')
    p.add_argument('--tasks', default=None, help='only these task ids, e.g. 0')
    p.add_argument('--allow-test-suite', action='store_true',
                   help='allow libero_10 (single-task runs that hold out initial states 0-9 instead)')
    args = p.parse_args()
    if args.lead < 0:
        p.error('--lead must be nonnegative')
    demo_ids = None
    if args.demo_ids:
        from glimmer_ppo.starts import parse_ids
        demo_ids = set(parse_ids(args.demo_ids, 1000000))
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[args.suite]()
    if args.suite == 'libero_10' and not args.allow_test_suite:
        raise SystemExit('libero_10 is the held-out test suite (pass --allow-test-suite for a single-task run)')
    keep = None if args.tasks is None else {int(t) for t in args.tasks.split(',')}
    by_name = {suite.get_task(i).name: i for i in range(suite.n_tasks) if keep is None or i in keep}
    rows = dict(task_id=[], demo=[], kind=[], step=[], state=[], settle_gripper=[])
    for path in sorted(args.demos.glob('*_demo.hdf5')):
        name = path.name.removesuffix('_demo.hdf5')
        if name not in by_name:
            print('skip (not in suite):', name)
            continue
        with h5py.File(path, 'r') as f:
            demos = sorted(f['data'].keys(), key=lambda k: int(k.split('_')[1]))
            if demo_ids is not None:
                demos = [k for k in demos if int(k.split('_')[1]) in demo_ids]
            demos = demos[:args.max_demos]
            for key in demos:
                states = f['data'][key]['states'][()]
                gripper = f['data'][key]['actions'][()][:, -1]
                for step, kind, grip in start_events(gripper, args.lead, args.final_release_only):
                    rows['task_id'].append(by_name[name]); rows['demo'].append(int(key.split('_')[1]))
                    rows['kind'].append('pre_' + kind); rows['step'].append(step); rows['state'].append(states[step])
                    rows['settle_gripper'].append(grip)
        print(f'{name}: {sum(t == by_name[name] for t in rows["task_id"])} starts')
    # MuJoCo state length differs between scenes (different objects), so pad
    # with NaN and keep each row's true length.
    if not rows['state']:
        raise SystemExit('No matching demo starts found; check --demos, --suite, --tasks and --demo-ids')
    lengths = np.array([len(s) for s in rows['state']])
    padded = np.full((len(lengths), lengths.max()), np.nan)
    for i, s in enumerate(rows['state']):
        padded[i, :len(s)] = s
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, task_id=np.array(rows['task_id']), demo=np.array(rows['demo']),
                        kind=np.array(rows['kind']), step=np.array(rows['step']), state=padded, length=lengths,
                        settle_gripper=np.array(rows['settle_gripper']), suite=np.array(args.suite),
                        lead=np.array(args.lead))
    print(f'saved {len(rows["task_id"])} starts to {args.output}')


if __name__ == '__main__':
    main()
