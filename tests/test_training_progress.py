import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import torch
from torch import nn

from glimmer_ppo.policy import Policy


def test_training_prints_phases_and_saves_checkpoint_without_gpu(tmp_path, monkeypatch, capsys):
    # Reproduce paper-only environments where the legacy harness is absent.
    monkeypatch.setitem(sys.modules, 'glimmer_ppo.env', None)
    monkeypatch.setitem(sys.modules, 'llm_agent_memory.harness', None)
    scope = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/train_ppo.py'))
    main = scope['main']
    globals_ = main.__globals__

    class FakePolicy(Policy):
        def __init__(self, cfg):
            nn.Module.__init__(self)
            self.cfg = cfg
            self.model = nn.Linear(2, 2)
            self.value_head = nn.Linear(2, 1)
            self.n_lora_targets = 1

        def act(self, messages, tools=None):
            return 'done', False, dict(inputs={'input_ids': torch.tensor([[1, 2]])},
                                       response=torch.tensor([3, 4]))

        def score(self, record, with_value=True, reference=False):
            h = self.model(torch.ones(2))
            return h.log_softmax(0), self.value_head(h).squeeze() if with_value else None

        def save(self, path):
            path.mkdir(parents=True)
            (path / 'saved.txt').write_text('saved')

    class FakeEnv:
        tools = None
        start = SimpleNamespace(label='init10', task_id=0)

        def __init__(self, *args):
            pass

        def reset(self, start):
            return []

        def step(self, text, truncated):
            return .1, True, dict(status='ok', tool='done', steps=1, success=False,
                                  decisions=1, goal_fraction=0., reward_parts={})

        def close(self):
            pass

    monkeypatch.setitem(globals_, 'Policy', FakePolicy)
    monkeypatch.setitem(globals_, 'PaperEnv', FakeEnv)
    monkeypatch.setitem(globals_, 'StartSampler', lambda *a, **k: lambda: None)
    monkeypatch.setattr('sys.argv', ['train_ppo.py', '--output', str(tmp_path), '--updates', '1',
                                   '--decisions-per-update', '2', '--epochs', '1', '--minibatch', '1'])
    main()
    output = capsys.readouterr()
    assert 'Rollout decisions' in output.err
    assert 'PPO backward passes' in output.err
    assert 'PPO updates' in output.err
    for phase in ('generating reply', 'scoring policy and reference', 'reward=+0.1000',
                  'computing advantages', 'optimizing', 'checkpoint saved', 'training complete'):
        assert phase in output.out
    assert (tmp_path / 'checkpoints/update_0001/saved.txt').exists()
    assert json.loads((tmp_path / 'metrics.jsonl').read_text())['update'] == 1
    assert (tmp_path / 'train.pid').read_text().strip().isdigit()
