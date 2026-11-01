from dataclasses import asdict
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest
import torch
from torch import nn

from glimmer_ppo.paper_env import PaperEnvConfig
from glimmer_ppo.policy import PolicyConfig


def test_frozen_comparison_pairs_starts_and_seeds_without_optimizer(tmp_path, monkeypatch):
    config = dict(args={'env': 'paper', 'task': '0'},
                  policy=asdict(PolicyConfig(load_in_4bit=True)), env=asdict(PaperEnvConfig(max_calls=8)))
    checkpoints = [tmp_path / 'update_0011', tmp_path / 'update_0012']
    for path in checkpoints:
        path.mkdir()
        (path / 'training_config.json').write_text(json.dumps(config))
    states = tmp_path / 'states.npz'
    np.savez(states, task_id=[0, 0], demo=[10, 11], kind=['pre_release', 'pre_release'],
             step=[1, 2], state=[[1., 2.], [3., 4.]], length=[2, 2],
             settle_gripper=[1., 1.], suite=np.array('libero_10'))
    instances, resets, loaded = [], [], []

    class FrozenPolicy(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.current = None
            instances.append(self)

        def load(self, path):
            self.current = path.name
            loaded.append(path.name)

        def act(self, messages, tools):
            assert not torch.is_grad_enabled()
            assert not self.training
            assert not self.weight.requires_grad
            return 'reply', False, {'response': [1, 2]}

    class Env:
        tools = None
        messages = []

        def __init__(self, path, cfg):
            self.progress = {'fraction': .5, 'grasped_objects_of_interest': ['tomato_sauce_1']}

        def reset(self, start):
            self.start = start
            resets.append((start.label, start.sim_state.tolist(), torch.initial_seed()))
            return []

        def step(self, text, truncated):
            success = instances[0].current == 'update_0012' or self.start.sim_state[0] == 1
            return .1, True, dict(success=success, steps=1, status='success' if success else 'call_budget',
                                  outcome='success' if success else 'call_budget', tool='move_to',
                                  goal_fraction=1. if success else .5)

        def close(self):
            pass

    def forbidden_optimizer(*a, **k):
        raise AssertionError('Evaluation must not create an optimizer')

    scope = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/compare_demo_checkpoints.py'))
    main = scope['main']
    monkeypatch.setitem(main.__globals__, 'Policy', FrozenPolicy)
    monkeypatch.setitem(main.__globals__, 'PaperEnv', Env)
    monkeypatch.setattr(torch.optim, 'AdamW', forbidden_optimizer)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    out = tmp_path / 'comparison'
    out.mkdir()
    (out / 'compare.log').touch()  # tee may create the terminal log before Python starts.
    monkeypatch.setattr(sys, 'argv', ['compare', '--checkpoints', *map(str, checkpoints),
                                    '--demo-states', str(states), '--output', str(out), '--samples', '2'])
    main()
    assert len(instances) == 1
    assert loaded == ['update_0011', 'update_0012']
    assert resets[:4] == resets[4:]
    assert instances[0].weight.grad is None
    torch.testing.assert_close(instances[0].weight, torch.ones(1))
    summary = json.loads((out / 'summary.json').read_text())
    assert [c['success_rate'] for c in summary['checkpoints']] == [.5, 1.]
    assert summary['paired']['second_only_success'] == 2
    assert summary['paired']['both_success'] == 2
    assert summary['paired']['trials'] == 4
    assert len((out / 'results.jsonl').read_text().splitlines()) == 8
    assert json.loads((out / 'progress.json').read_text()) == {
        'completed': 8, 'total': 8, 'learning': False, 'phase': 'complete'}
    with pytest.raises(SystemExit):
        main()  # Never overwrite an existing evaluation.
    assert len(instances) == 1
