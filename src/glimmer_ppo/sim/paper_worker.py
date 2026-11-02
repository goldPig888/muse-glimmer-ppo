"""The paper's LIBERO worker, plus reward information.

This is ``inspect_robots_libero/_worker.py`` from LLM_LIBERO (commit 7db834a)
with optional reward/start/assistance support: same 256x256 cameras, same
agentview flip, same rot6d handling, same 10 settling steps.

* every reply carries ``progress`` (BDDL goal predicates, grasp of an object
  of interest, lift height). Privileged: used for the reward only, the
  observation the model sees is unchanged.
* ``reset`` may take ``sim_state`` (a flattened MuJoCo state from a demo) and
  ``settle_gripper`` (+1 keeps the hand closed while settling).
* ``get_state`` returns the current flattened MuJoCo state.
* opt-in ``object_assistance`` installs bounded reference-tracking forces on
  target objects; disabled for ordinary runs and exactly zero at gain zero.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

wire = sys.stdout
sys.stdout = sys.stderr
_config = os.path.join(os.environ.get('LIBERO_CONFIG_PATH', os.path.expanduser('~/.libero')), 'config.yaml')
if not os.path.exists(_config):
    wire.write(json.dumps({'error': f'LIBERO is not initialised ({_config} missing). '
                           'Run python -c "import libero.libero" once and answer N.'}) + '\n')
    wire.flush()
    sys.exit(1)
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
import robosuite.utils.transform_utils as T  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
from glimmer_ppo.assistance import AssistanceConfig, ObjectAssistant  # noqa: E402

LIFT_M = 0.03


def object_heights(domain, names):
    heights = {}
    for name in names:
        try:
            heights[name] = float(domain.sim.data.body_xpos[domain.obj_body_id[name]][2])
        except (AttributeError, KeyError, IndexError):
            pass
    return heights


def _position(domain, name):
    """World position of an object body, or of a region site (e.g. basket_1_contain_region)."""
    try:
        if name in domain.obj_body_id:
            return np.asarray(domain.sim.data.body_xpos[domain.obj_body_id[name]], dtype=float)
        site = domain.object_sites_dict[name]
        return np.asarray(domain.sim.data.get_site_xpos(site.name), dtype=float)
    except (AttributeError, KeyError, ValueError):
        return None


def progress(env, start_heights):
    """Reward information. Objects to move = first argument of two-argument goals
    (In/On); their destination = the second argument (a region or an object)."""
    domain = env.env
    goals = domain.parsed_problem['goal_state']
    satisfied = [bool(domain._eval_predicate(g)) for g in goals]
    moves = [(str(g[1]), str(g[2]), ok) for g, ok in zip(goals, satisfied) if len(g) == 3]
    to_move = {a for a, _, _ in moves}
    grasped = []
    for name in to_move:
        obj = getattr(domain, 'objects_dict', {}).get(name)
        if obj is None:
            continue
        try:
            if domain._check_grasp(gripper=domain.robots[0].gripper, object_geoms=obj):
                grasped.append(name)
        except Exception:
            pass
    heights = object_heights(domain, grasped)
    lifted = [n for n in grasped if n in start_heights and heights.get(n, -1) - start_heights[n] >= LIFT_M]
    # Dense distances (metres), for unsatisfied move goals only.
    hand = np.asarray(domain.robots[0].controller.ee_pos, dtype=float)
    reach, carry = None, None
    pending = [(a, b) for a, b, ok in moves if not ok]
    held = [(a, b) for a, b in pending if a in grasped]
    if held:
        a, b = held[0]
        pa, pb = _position(domain, a), _position(domain, b)
        if pa is not None and pb is not None:
            carry = float(np.linalg.norm(pa - pb))
    elif pending:
        dists = [np.linalg.norm(hand - pa) for a, _ in pending if (pa := _position(domain, a)) is not None]
        reach = float(min(dists)) if dists else None
    return {
        'goals': [' '.join(map(str, g)) for g in goals],
        'satisfied': satisfied,
        'fraction': float(np.mean(satisfied)) if satisfied else 0.0,
        'grasped_objects_of_interest': grasped,
        'lifted_objects_of_interest': lifted,
        'reach_m': reach,
        'carry_m': carry,
    }


def main():
    env = None
    video = None
    obs = None
    steps = 0
    start_heights = {}
    assistant = None

    def pack():
        robot = env.env.robots[0]
        measured = progress(env, start_heights)
        if assistant is not None:
            measured['voc'] = assistant.summary()
        return {
            'images': {k: (obs[k][::-1, ::-1] if k == 'agentview_image' else obs[k]).copy().tolist() for k in
                       ('agentview_image', 'robot0_eye_in_hand_image')},
            'eef_state': np.concatenate((robot.controller.ee_pos,
                robot.controller.ee_ori_mat[:, :2].T.reshape(-1),
                [np.clip((robot.gripper.current_action[-1] + 1.0) / 2.0, 0, 1)])).tolist(),
            'gripper_qpos': obs['robot0_gripper_qpos'].tolist(),
            'success': bool(env.check_success()), 'steps': steps,
            'progress': measured,
        }

    try:
        for line in sys.stdin:
            req = json.loads(line)
            try:
                if req['op'] == 'reset':
                    if assistant is not None:
                        assistant.close()
                        assistant = None
                    if env is not None:
                        env.close()
                    if video is not None:
                        video.close()
                    suite = benchmark.get_benchmark_dict()[req.get('suite', 'libero_10')]()
                    task_id = int(req['task_id'])
                    task = suite.get_task(task_id)
                    env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files')) /
                        task.problem_folder / task.bddl_file), camera_heights=256, camera_widths=256)
                    env.seed(int(req['env_seed']))
                    env.reset()
                    if req.get('sim_state') is not None:
                        obs = env.set_init_state(np.asarray(req['sim_state'], dtype=float))
                    else:
                        obs = env.set_init_state(suite.get_task_init_states(task_id)[int(req['init_state'])])
                    grip = float(req.get('settle_gripper', -1.0))
                    for _ in range(10):
                        obs, _, _, _ = env.step(np.array([0., 0., 0., 0., 0., 0., grip]))
                    env.env.robots[0].controller.use_delta = False
                    steps = 0
                    goals = env.env.parsed_problem['goal_state']
                    start_heights = object_heights(env.env, [str(g[1]) for g in goals if len(g) == 3])
                    if req.get('object_assistance') is not None:
                        if env.check_success():
                            raise ValueError('Assisted episode is already successful at its initial state')
                        settings = dict(req['object_assistance'])
                        reference = settings.pop('reference')
                        assistant = ObjectAssistant(env.env, reference, AssistanceConfig(**settings))
                    video = imageio.get_writer(req['video'], fps=20)
                    result = pack()
                    result['instruction'] = task.language
                elif req['op'] == 'step':
                    action = np.asarray(req['action'], dtype=float)
                    if action.shape != (10,) or not np.all(np.isfinite(action)):
                        raise ValueError('Invalid action')
                    a, b = action[3:6], action[6:9]
                    if np.linalg.norm(a) < 1e-6:
                        raise ValueError('Degenerate rot6d first column')
                    a = a / np.linalg.norm(a)
                    b = b - np.dot(a, b) * a
                    if np.linalg.norm(b) < 1e-6:
                        raise ValueError('Degenerate rot6d second column')
                    b = b / np.linalg.norm(b)
                    rotation = np.stack([a, b, np.cross(a, b)], axis=1)
                    action = np.concatenate([action[:3], T.quat2axisangle(T.mat2quat(rotation)), [2 * action[-1] - 1]])
                    if assistant is not None:
                        assistant.before_step(steps)
                    obs, _, _, _ = env.step(action)
                    if assistant is not None:
                        assistant.after_step(progress(env, start_heights)['grasped_objects_of_interest'])
                    steps += 1
                    result = pack()
                elif req['op'] == 'get_state':
                    wire.write(json.dumps({'sim_state': env.get_sim_state().tolist()}) + '\n')
                    wire.flush()
                    continue
                elif req['op'] == 'close':
                    break
                else:
                    raise ValueError('Unknown operation')
                video.append_data(np.concatenate([np.asarray(result['images'][k], dtype=np.uint8)
                    for k in ('agentview_image', 'robot0_eye_in_hand_image')], axis=1))
                wire.write(json.dumps(result) + '\n')
                wire.flush()
            except Exception as exc:
                wire.write(json.dumps({'error': type(exc).__name__ + ': ' + str(exc)}) + '\n')
                wire.flush()
    finally:
        if assistant is not None:
            assistant.close()
        if video is not None:
            video.close()
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
