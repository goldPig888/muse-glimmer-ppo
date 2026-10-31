import json
import random
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from glimmer_ppo.checkpoint import checkpoint_config, checkpoint_update, restore_checkpoint, save_checkpoint


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)

    def save(self, path):
        torch.save(self.state_dict(), path / 'weights.pt')

    def load(self, path):
        self.load_state_dict(torch.load(path / 'weights.pt', weights_only=True))


def test_checkpoint_restores_parameters_optimizer_and_random_states(tmp_path):
    p = TinyPolicy()
    opt = torch.optim.AdamW(p.parameters(), lr=.01)
    p.layer(torch.ones(2)).sum().backward()
    opt.step()
    sampler = SimpleNamespace(rng=random.Random(4))
    config = {'args': {'model': 'test'}}
    path = tmp_path / 'update_0002'
    save_checkpoint(p, opt, sampler, path, 2, config)
    expected = [t.clone() for t in p.parameters()]
    draws = random.random(), torch.rand(3), sampler.rng.random()
    for tensor in p.parameters():
        tensor.data.zero_()
    opt.state.clear()
    assert restore_checkpoint(p, opt, sampler, path) == 2
    assert random.random() == draws[0]
    torch.testing.assert_close(torch.rand(3), draws[1])
    assert sampler.rng.random() == draws[2]
    for tensor, reference in zip(p.parameters(), expected):
        torch.testing.assert_close(tensor, reference)
    assert len(opt.state) == 2
    assert checkpoint_config(path) == config
    assert checkpoint_update(path) == 2
    with pytest.raises(FileExistsError):
        save_checkpoint(p, opt, sampler, path, 2, config)


def test_legacy_checkpoint_uses_original_run_config_and_update_name(tmp_path):
    path = tmp_path / 'checkpoints/update_0002'
    path.mkdir(parents=True)
    (tmp_path / 'config.json').write_text(json.dumps({'args': {'task': '0'}}))
    assert checkpoint_update(path) == 2
    assert checkpoint_config(path)['args']['task'] == '0'


def test_failed_checkpoint_is_not_published(tmp_path):
    class BrokenPolicy:
        def save(self, path):
            (path / 'partial').write_text('partial')
            raise RuntimeError('disk failure')
    with pytest.raises(RuntimeError, match='disk failure'):
        save_checkpoint(BrokenPolicy(), None, None, tmp_path / 'update_0001', 1, {})
    assert not (tmp_path / 'update_0001').exists()
