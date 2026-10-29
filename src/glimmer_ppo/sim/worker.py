"""LIBERO worker for PPO: any suite, saved-state starts, goal progress.

Same line-delimited JSON protocol as ``llm_agent_memory.harness.worker`` (one
request per stdin line, one reply per stdout line), with three changes:

* ``reset`` takes ``suite`` (default ``libero_90``) instead of always LIBERO-10,
  and may take ``sim_state`` (a flattened MuJoCo state, e.g. from a demo) to
  start mid-task instead of from an initial state.
* every reply carries ``progress``: which BDDL goal predicates hold, and
  whether the gripper is grasping one of the task's objects of interest.
  These are privileged and used only for the reward, never shown to the model.
* ``get_state`` returns the current flattened MuJoCo state.
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
from llm_agent_memory.harness.worker_support import calibrate, check_targets  # noqa: E402


def goal_progress(env):
    """Fraction of BDDL goal predicates satisfied, plus a grasp check.

    LIBERO's own success check is the conjunction of these predicates
    (BDDLBaseDomain._check_success), so ``fraction == 1`` iff success.
    """
    domain = env.env
    goals = domain.parsed_problem['goal_state']
    satisfied = [bool(domain._eval_predicate(g)) for g in goals]
    grasped = []
    for name in getattr(domain, 'obj_of_interest', []) or []:
        obj = domain.objects_dict.get(name) if hasattr(domain, 'objects_dict') else None
        if obj is None:
            continue
        try:
            if domain._check_grasp(gripper=domain.robots[0].gripper, object_geoms=obj):
                grasped.append(name)
        except Exception:  # object without collision geoms, fixtures
            pass
    return {
        'goals': [' '.join(map(str, g)) for g in goals],
        'satisfied': satisfied,
        'fraction': float(np.mean(satisfied)) if satisfied else 0.0,
        'grasped_objects_of_interest': grasped,
    }


def main():
    env = None
    video = None
    obs = None
    steps = 0

    def pack():
        robot = env.env.robots[0]
        return {
            **calibrate(env, obs),
            'eef_state': np.concatenate((robot.controller.ee_pos,
                robot.controller.ee_ori_mat[:, :2].T.reshape(-1),
                [np.clip((robot.gripper.current_action[-1] + 1.0) / 2.0, 0, 1)])).tolist(),
            'gripper_qpos': obs['robot0_gripper_qpos'].tolist(),
            'success': bool(env.check_success()),
            'steps': steps,
            'progress': goal_progress(env),
        }

    try:
        for line in sys.stdin:
            req = json.loads(line)
            try:
                if req['op'] == 'reset':
                    if env is not None:
                        env.close()
                    if video is not None:
                        video.close()
                        video = None
                    suite = benchmark.get_benchmark_dict()[req.get('suite', 'libero_90')]()
                    task_id = int(req['task_id'])
                    task = suite.get_task(task_id)
                    env = OffScreenRenderEnv(
                        bddl_file_name=str(Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file),
                        camera_heights=512, camera_widths=512, camera_depths=True)
                    env.seed(int(req['env_seed']))
                    env.reset()
                    if req.get('sim_state') is not None:
                        obs = env.set_init_state(np.asarray(req['sim_state'], dtype=float))
                    else:
                        obs = env.set_init_state(suite.get_task_init_states(task_id)[int(req['init_state'])])
                    # Ten neutral steps let objects settle; same as the harness worker.
                    # A start saved while holding an object settles with the hand closed.
                    grip = float(req.get('settle_gripper', -1.0))
                    for _ in range(10):
                        obs, _, _, _ = env.step(np.array([0., 0., 0., 0., 0., 0., grip]))
                    env.env.robots[0].controller.use_delta = False
                    steps = 0
                    if req.get('video'):
                        video = imageio.get_writer(req['video'], fps=20)
                    result = pack()
                    result['instruction'] = task.language
                    result['task_name'] = task.name
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
                    obs, _, _, _ = env.step(action)
                    steps += 1
                    result = pack()
                elif req['op'] == 'check':
                    result = check_targets(env, req['targets'], req.get('fingers_may_touch', False), req.get('carry', False))
                    wire.write(json.dumps(result) + '\n')
                    wire.flush()
                    continue
                elif req['op'] == 'get_state':
                    wire.write(json.dumps({'sim_state': env.get_sim_state().tolist()}) + '\n')
                    wire.flush()
                    continue
                elif req['op'] == 'close':
                    break
                else:
                    raise ValueError('Unknown operation')
                if video is not None:
                    video.append_data(np.concatenate([np.asarray(result['images'][k], dtype=np.uint8)
                        for k in ('agentview_image', 'robot0_eye_in_hand_image')], axis=1))
                wire.write(json.dumps(result) + '\n')
                wire.flush()
            except Exception as exc:
                wire.write(json.dumps({'error': type(exc).__name__ + ': ' + str(exc)}) + '\n')
                wire.flush()
    finally:
        if video is not None:
            video.close()
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
