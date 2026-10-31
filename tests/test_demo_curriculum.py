from pathlib import Path
import runpy

import numpy as np
import pytest

from glimmer_ppo.starts import StartSampler
from glimmer_ppo.toolcall import parse_tool_call

scripts = Path(__file__).parents[1] / 'scripts'
extract = runpy.run_path(str(scripts / 'extract_start_states.py'))
check = runpy.run_path(str(scripts / 'check_demo_starts.py'))


def test_final_release_uses_previous_gripper_command_at_restored_state():
    gripper = np.array([-1, -1, 1, 1, -1, -1, 1, 1, -1])
    assert extract['start_events'](gripper, 0, True) == [(8, 'release', 1.)]
    assert extract['start_events'](gripper, 1, True) == [(7, 'release', 1.)]
    assert extract['start_events'](gripper, 2, True) == [(6, 'release', -1.)]
    assert extract['start_events'](gripper, 100, True) == [(0, 'release', -1.)]
    assert extract['start_events'](np.ones(5), 1, True) == []
    with pytest.raises(ValueError):
        extract['start_events'](gripper, -1)


def fixture_states(path):
    np.savez(path, task_id=[0, 0], demo=[10, 11], kind=['pre_grasp', 'pre_release'],
             step=[1, 2], state=[[1., 2.], [3., 4.]], length=[2, 2],
             settle_gripper=[-1., -1.], suite=np.array('libero_10'))


def test_curriculum_sampler_only_uses_selected_states_and_recorded_gripper(tmp_path):
    path = tmp_path / 'states.npz'
    fixture_states(path)
    sampler = StartSampler('libero_10', [0], [10], path, 1, kinds=('pre_release',), allow_test_suite=True)
    for _ in range(5):
        start = sampler()
        assert start.label == 'demo11_pre_release_2'
        assert start.settle_gripper == -1.  # earlier offsets may be before the grasp
        np.testing.assert_equal(start.sim_state, [3., 4.])
    with pytest.raises(ValueError, match='refusing to fall back'):
        StartSampler('libero_10', [1], [10], path, 1, allow_test_suite=True)
    with pytest.raises(ValueError, match='suite does not match'):
        StartSampler('libero_90', [0], [10], path, 1)


def test_validation_rejects_solved_or_unusable_easy_starts():
    reject = check['rejection_reason']
    good = dict(fraction=.5, grasped_objects_of_interest=['tomato_sauce_1'])
    assert reject(good, False, .5, True) is None
    assert 'already successful' in reject(dict(good, fraction=1), True, .5, True)
    assert 'not enough goal' in reject(dict(good, fraction=0), False, .5, True)
    assert 'not held' in reject(dict(good, grasped_objects_of_interest=[]), False, .5, True)


def test_scripted_release_baseline_does_not_move_the_arm():
    class Env:
        def step(self, text):
            name, args = parse_tool_call(text)
            assert name == 'move_to'
            assert args['targets'] == {'gripper': 0.0}
            return 1.2, True, dict(success=True, status='success', goal_fraction=1.)
    result = check['release_in_place'](Env())
    assert result['release_success']
    assert result['release_goal_fraction'] == 1.


def test_release_retreat_baseline_opens_before_moving_and_stops_on_success():
    from types import SimpleNamespace
    class Env:
        embodiment = SimpleNamespace(last={'eef_state': [0, 0, .6]})
        calls = []

        def step(self, text):
            _, args = parse_tool_call(text)
            self.calls.append(args['targets'])
            success = len(self.calls) == 5
            return .1, success, dict(success=success, status='success' if success else 'ok',
                                     goal_fraction=1. if success else .5)
    env = Env()
    result = check['release_in_place'](env, max_calls=8, retreat_m=.05)
    assert result['release_success']
    assert result['release_calls'] == 5
    assert env.calls[:2] == [{'gripper': 0.}, {'gripper': 0.}]
    assert env.calls[2]['z'] == pytest.approx(.65)
    assert len(env.calls) == 5
