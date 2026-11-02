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
            path.mkdir(parents=True, exist_ok=True)
            (path / 'saved.txt').write_text('saved')
            torch.save(self.state_dict(), path / 'weights.pt')

        def load(self, path):
            self.load_state_dict(torch.load(path / 'weights.pt', weights_only=True))

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
    first_state = torch.load(tmp_path / 'checkpoints/update_0001/training_state.pt', weights_only=True)
    resumed = tmp_path / 'resumed'
    monkeypatch.setattr('sys.argv', ['train_ppo.py', '--output', str(resumed), '--resume',
                                   str(tmp_path / 'checkpoints/update_0001'), '--updates', '2'])
    main()
    resumed_output = capsys.readouterr()
    assert 'next update=2' in resumed_output.out
    assert 'restored optimizer' in resumed_output.out
    assert json.loads((resumed / 'metrics.jsonl').read_text())['update'] == 2
    second_state = torch.load(resumed / 'checkpoints/update_0002/training_state.pt', weights_only=True)
    first_step = next(iter(first_state['optimizer']['state'].values()))['step']
    second_step = next(iter(second_state['optimizer']['state'].values()))['step']
    assert second_step == first_step * 2

    # Older adapters/value heads can continue without optimizer state.
    (tmp_path / 'checkpoints/update_0001/training_state.pt').unlink()
    legacy = tmp_path / 'legacy_resumed'
    monkeypatch.setattr('sys.argv', ['train_ppo.py', '--output', str(legacy), '--resume',
                                   str(tmp_path / 'checkpoints/update_0001'), '--updates', '2'])
    main()
    assert 'optimizer and RNG states were not saved and start fresh' in capsys.readouterr().out
    legacy_state = torch.load(legacy / 'checkpoints/update_0002/training_state.pt', weights_only=True)
    assert next(iter(legacy_state['optimizer']['state'].values()))['step'] == first_step

    # Teacher/SFT weights initialize a new PPO experiment, not an old optimizer.
    initialized = tmp_path / 'initialized'
    monkeypatch.setattr('sys.argv', ['train_ppo.py', '--output', str(initialized), '--init-policy',
                                   str(tmp_path / 'checkpoints/update_0001'), '--updates', '1',
                                   '--decisions-per-update', '2', '--epochs', '1', '--minibatch', '1'])
    main()
    assert 'fresh optimizer, update counter and critic warmup' in capsys.readouterr().out
    initialized_state = torch.load(initialized / 'checkpoints/update_0001/training_state.pt', weights_only=True)
    assert initialized_state['update'] == 1
    assert next(iter(initialized_state['optimizer']['state'].values()))['step'] == first_step

    from glimmer_ppo.assistance import AssistanceConfig, AssistanceCurriculum
    import random
    class FakeAssistedSampler:
        config = AssistanceConfig()
        manifest_hash = 'fixture'
        rng = random.Random(1)
        curriculum = AssistanceCurriculum(window=1)

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self):
            return SimpleNamespace(label='full_demo_start', task_id=0,
                                   gain=self.curriculum.gain)

        def state_dict(self):
            return dict(curriculum=self.curriculum.state_dict())

    class FakeAssistedEnv(FakeEnv):
        def reset(self, start):
            self.start = start
            return []

        def step(self, text, truncated):
            reward, done, info = super().step(text, truncated)
            info['voc'] = dict(gain=self.start.gain, assisted=True, object_score=.9,
                               motion_score=.9, contact_score=.9, force_rms_N=1., impulse_Ns=.1)
            info['success'] = True
            return reward, done, info
    monkeypatch.setitem(globals_, 'AssistedStartSampler', FakeAssistedSampler)
    monkeypatch.setitem(globals_, 'PaperEnv', FakeAssistedEnv)
    assisted = tmp_path / 'assisted'
    monkeypatch.setattr('sys.argv', ['train_ppo.py', '--output', str(assisted), '--updates', '1',
                                   '--object-references', str(tmp_path), '--decisions-per-update', '2',
                                   '--epochs', '1', '--minibatch', '1'])
    main()
    assisted_state = torch.load(assisted / 'checkpoints/update_0001/training_state.pt', weights_only=True)
    assert assisted_state['sampler_state']['curriculum']['gain'] == .8
    assisted_metrics = json.loads((assisted / 'metrics.jsonl').read_text())
    assert assisted_metrics['assisted_episodes'] == 2
    assert assisted_metrics['assistance_next_gain'] == .8
