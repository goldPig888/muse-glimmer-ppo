"""Extract full object/hand trajectories for a DexMachina-style curriculum.

These are FULL demonstration starts, not reverse-curriculum mid-task starts.
Demo ids are not LIBERO benchmark init-state ids. No model is loaded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm.auto import tqdm

from glimmer_ppo.assistance import load_reference
from glimmer_ppo.starts import parse_ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--demos', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--demo-ids', default='10-14')
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Use a fresh reference output directory')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/glimmer_ppo_numba')
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    suite = benchmark.get_benchmark_dict()['libero_10']()
    task = suite.get_task(args.task)
    source = args.demos / f'{task.name}_demo.hdf5'
    env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files')) /
                            task.problem_folder / task.bddl_file), camera_heights=32, camera_widths=32)
    manifest = dict(version=1, suite='libero_10', task_id=args.task, task_name=task.name,
                    source=str(source.resolve()), start_kind='full_demo_initial_state',
                    note='Demo ids are not benchmark initial-state ids. Panda adaptation of virtual '
                         'object controllers with end-effector motion and binary grasp guidance.', references=[])
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(source) as data:
            for demo in tqdm(parse_ids(args.demo_ids, 1000000), desc='Extract full references', unit='demo'):
                group = data[f'data/demo_{demo}']
                states = group['states'][()]
                actions = group['actions'][()]
                env.seed(7000 + demo)
                env.reset()
                env.set_init_state(states[0])
                domain = env.env
                goals = domain.parsed_problem['goal_state']
                names = sorted({str(g[1]) for g in goals if len(g) == 3})
                if not names or any(domain._eval_predicate(g) for g in goals):
                    raise ValueError(f'demo_{demo} is not an unsolved full-task start')
                bodies = [domain.obj_body_id[n] for n in names]
                site = domain.robots[0].eef_site_id
                positions, matrices, hands, hand_matrices, grasps = [], [], [], [], []
                for state in states:
                    # Reading recorded states is not rollout playback. Avoid
                    # rendering every frame; forward recomputes geometry/contact.
                    env.set_state(state)
                    domain.sim.forward()
                    simdata = domain.sim.data
                    positions.append(simdata.body_xpos[bodies].copy())
                    matrices.append(simdata.body_xmat[bodies].reshape(-1, 3, 3).copy())
                    hands.append(simdata.site_xpos[site].copy())
                    hand_matrices.append(simdata.site_xmat[site].reshape(3, 3).copy())
                    grasps.append([bool(domain._check_grasp(gripper=domain.robots[0].gripper,
                        object_geoms=domain.objects_dict[n])) for n in names])
                if not env.check_success():
                    raise ValueError(f'demo_{demo} final state does not satisfy the task')
                if any(grasps[0]):
                    raise ValueError(f'demo_{demo} starts already grasping an object; use a full-task start')
                destination = args.output / f'demo_{demo:04d}.npz'
                np.savez_compressed(destination, task_id=args.task, demo_id=demo,
                    body_names=np.array(names), initial_state=states[0], settle_gripper=-1.,
                    object_pos=positions, object_mat=matrices, eef_pos=hands, eef_mat=hand_matrices,
                    grasp=np.array(grasps, dtype=bool), demo_actions=actions)
                load_reference(destination)
                manifest['references'].append(dict(file=destination.name, demo_id=demo,
                    frames=len(states), grasp_frames=int(np.any(grasps, axis=1).sum()),
                    sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))
                tqdm.write(f'[references] demo={demo} frames={len(states)} verified final success')
        (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    finally:
        env.close()
    print(f'Saved {len(manifest["references"])} full-task references to {args.output}', flush=True)


if __name__ == '__main__':
    main()
