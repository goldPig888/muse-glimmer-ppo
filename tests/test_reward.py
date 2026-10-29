import json
import math

import numpy as np
import pytest

from glimmer_ppo.reward import RewardConfig, decision_reward, status_of
from glimmer_ppo.starts import StartSampler, parse_ids


@pytest.mark.parametrize('feedback,status', [
    (json.dumps({'status': 'completed', 'executed': []}), 'completed'),
    (json.dumps({'status': 'COLLISION'}), 'COLLISION'),
    (json.dumps({'status': 'grasp_failed'}), 'grasp_failed'),
    ('plan_error: No curl command found.', 'plan_error'),
    ('Response truncated; nothing executed.', 'truncated'),
    ('New observation requested; no motion.', 'observation'),
    ('observation_limit: propose a pixel target', 'observation_limit'),
])
def test_status_of(feedback, status):
    assert status_of(feedback) == status


def test_success_gets_bonus_and_progress():
    r = decision_reward(RewardConfig(), prev_fraction=0.0, fraction=1.0, success=True, first_grasp=False,
                        status='success', repeated=False)
    assert math.isclose(r.total, 1.0 + 0.5 - 0.005)


def test_rejection_and_repeat_penalties():
    r = decision_reward(RewardConfig(), prev_fraction=0.0, fraction=0.0, success=False, first_grasp=False,
                        status='COLLISION', repeated=True)
    assert math.isclose(r.total, -0.05 - 0.1 - 0.005)


def test_physical_failure_is_not_a_rejection():
    r = decision_reward(RewardConfig(), prev_fraction=0.0, fraction=0.0, success=False, first_grasp=False,
                        status='grasp_failed', repeated=False)
    assert r.parts['rejected'] == 0.0


def test_undoing_progress_is_penalized():
    r = decision_reward(RewardConfig(), prev_fraction=0.5, fraction=0.0, success=False, first_grasp=False,
                        status='completed', repeated=False)
    assert r.parts['progress'] == -0.25


def test_parse_ids_and_held_out_suite():
    assert parse_ids('0-2,5', 90) == [0, 1, 2, 5]
    with pytest.raises(ValueError):
        parse_ids('95', 90)
    with pytest.raises(ValueError):
        StartSampler('libero_10')


def test_demo_starts_are_unpadded(tmp_path):
    path = tmp_path / 'starts.npz'
    state = np.full((2, 5), np.nan)
    state[0, :3] = [1, 2, 3]
    state[1, :5] = [1, 2, 3, 4, 5]
    np.savez(path, task_id=np.array([0, 1]), demo=np.array([0, 0]), kind=np.array(['pre_grasp', 'pre_release']),
             step=np.array([10, 40]), state=state, length=np.array([3, 5]))
    sampler = StartSampler('libero_90', tasks=[0, 1], demo_states=path, demo_fraction=1.0)
    by_task = {s.task_id: s for s in sampler.demo}
    assert list(by_task[0].sim_state) == [1, 2, 3]
    assert by_task[0].settle_gripper == -1.0
    assert by_task[1].settle_gripper == 1.0      # starts holding: settle with the hand closed
    assert sampler().sim_state is not None
