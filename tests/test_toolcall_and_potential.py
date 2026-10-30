import math

import pytest

from glimmer_ppo.reward import PotentialConfig, potential, potential_reward
from glimmer_ppo.toolcall import parse_tool_call

MOVE = {'targets': {'x': 0.1, 'z': 0.9}, 'note': 'reach'}


@pytest.mark.parametrize('text', [
    '<tool_call>{"name": "move_to", "arguments": {"targets": {"x": 0.1, "z": 0.9}, "note": "reach"}}</tool_call>',
    'thinking... {"name": "move_to", "arguments": {"targets": {"x": 0.1, "z": 0.9}, "note": "reach"}}',
    '{"type": "function", "function": {"name": "move_to", "arguments": "{\\"targets\\": {\\"x\\": 0.1, \\"z\\": 0.9}, \\"note\\": \\"reach\\"}"}}',
    'move_to({"targets": {"x": 0.1, "z": 0.9}, "note": "reach"})',
])
def test_parses_common_formats(text):
    assert parse_tool_call(text) == ('move_to', MOVE)


def test_takes_last_call_and_rejects_other_tools():
    text = ('draft: {"name": "move_to", "arguments": {"targets": {"z": 1.0}, "note": "a"}} '
            'final: {"name": "done", "arguments": {"summary": "ok"}}')
    assert parse_tool_call(text) == ('done', {'summary': 'ok'})
    assert parse_tool_call('{"name": "rm_rf", "arguments": {}}') is None
    assert parse_tool_call('no call here') is None


def test_potential_terms():
    cfg = PotentialConfig()
    empty = dict(fraction=0.0, grasped_objects_of_interest=[], lifted_objects_of_interest=[])
    held = dict(fraction=0.0, grasped_objects_of_interest=['alphabet_soup_1'], lifted_objects_of_interest=['alphabet_soup_1'])
    half = dict(fraction=0.5, grasped_objects_of_interest=[], lifted_objects_of_interest=[])
    assert potential(cfg, empty) == 0.0
    assert math.isclose(potential(cfg, held), 0.3)
    assert math.isclose(potential(cfg, half), 0.5)


def test_grab_and_drop_nets_zero_shaping():
    cfg = PotentialConfig()
    empty = dict(fraction=0.0, grasped_objects_of_interest=[], lifted_objects_of_interest=[])
    held = dict(fraction=0.0, grasped_objects_of_interest=['x'], lifted_objects_of_interest=['x'])
    up = potential_reward(cfg, prev_progress=empty, progress=held, success=False, invalid=False, repeated=False)
    down = potential_reward(cfg, prev_progress=held, progress=empty, success=False, invalid=False, repeated=False)
    assert math.isclose(up.parts['shaping'] + down.parts['shaping'], 0.0)


def test_placing_first_can_in_basket_is_net_positive():
    # releasing the soup into the basket loses the hold/lift potential but meets 1 of 2 goal conditions
    cfg = PotentialConfig()
    held = dict(fraction=0.0, grasped_objects_of_interest=['soup'], lifted_objects_of_interest=['soup'])
    placed = dict(fraction=0.5, grasped_objects_of_interest=[], lifted_objects_of_interest=[])
    r = potential_reward(cfg, prev_progress=held, progress=placed, success=False, invalid=False, repeated=False)
    assert math.isclose(r.parts['shaping'], 0.5 - 0.3)


def test_sparse_ablation_and_costs():
    cfg = PotentialConfig(shaping=False)
    p = dict(fraction=1.0, grasped_objects_of_interest=[], lifted_objects_of_interest=[])
    r = potential_reward(cfg, prev_progress={}, progress=p, success=True, invalid=False, repeated=False)
    assert r.parts['shaping'] == 0.0 and math.isclose(r.total, 1.0 - 0.005)
    r = potential_reward(PotentialConfig(), prev_progress={}, progress={}, success=False, invalid=True, repeated=True)
    assert math.isclose(r.total, -0.05 - 0.1 - 0.005)
