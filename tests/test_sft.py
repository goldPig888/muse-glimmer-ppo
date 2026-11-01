import json
from pathlib import Path
import runpy
import sys

import pytest
import torch
from torch import nn

from glimmer_ppo.policy import Policy, PolicyConfig


def test_supervision_masks_prompt_and_checks_native_template_prefix():
    class Processor:
        bad = False

        def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
            ids = [1, 2, 3] if add_generation_prompt else [1, 2, 3, 4, 5]
            if self.bad and not add_generation_prompt:
                ids[0] = 8
            return {'input_ids': torch.tensor([ids]), 'pixel_values': torch.ones(1, 3)}

    policy = Policy.__new__(Policy)
    nn.Module.__init__(policy)
    policy.cfg = PolicyConfig()
    policy.processor = Processor()
    rec = policy.supervised_record([{'role': 'user', 'content': 'camera'}],
                                   {'role': 'assistant', 'content': 'action'})
    assert rec['inputs']['input_ids'].tolist() == [[1, 2, 3]]
    assert rec['response'].tolist() == [4, 5]
    assert 'pixel_values' in rec['inputs']
    policy.processor.bad = True
    with pytest.raises(ValueError, match='prefix'):
        policy.supervised_record([], {'role': 'assistant', 'content': 'action'})


def test_sft_trains_response_likelihood_freezes_critic_and_saves_weights(tmp_path, monkeypatch):
    script = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/train_sft.py'))
    main = script['main']
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    rows = [dict(split=split, source={'task_id': 0, 'init_state': state}, messages=[],
                 target={'role': 'assistant', 'content': 'action'}, tools=[], call=i)
            for split, state in [('train', 0), ('validation', 8)] for i in range(3)]
    (dataset / 'examples.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    instances = []

    class TinyPolicy(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.model = nn.Linear(1, 2, bias=False)
            self.model.weight.data.zero_()
            self.value_head = nn.Linear(1, 1)
            self.value_head.weight.data.fill_(.125)
            self.cfg = cfg
            instances.append(self)

        def supervised_record(self, messages, target, tools):
            return {}

        def score(self, rec, with_value):
            assert not with_value
            return self.model(torch.ones(1)).log_softmax(0)[1:2], None

        def save(self, destination):
            torch.save(self.state_dict(), destination / 'weights.pt')

    monkeypatch.setitem(main.__globals__, 'Policy', TinyPolicy)
    out = tmp_path / 'run'
    monkeypatch.setattr(sys, 'argv', ['train_sft', '--dataset', str(dataset), '--output', str(out),
                                    '--epochs', '2', '--lr', '.1', '--accumulate', '2'])
    main()
    metrics = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
    assert metrics[-1]['validation_nll'] < metrics[0]['validation_nll']
    assert instances[0].value_head.weight.item() == .125
    assert instances[0].value_head.weight.grad is None
    assert (out / 'epoch_0002/training_config.json').exists()
    assert not (out / 'epoch_0002/training_state.pt').exists()
    assert json.loads((out / 'progress.json').read_text())['phase'] == 'complete'
    rows[-1]['source']['init_state'] = 0
    (dataset / 'examples.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    with pytest.raises(ValueError, match='disjoint'):
        script['load_examples'](dataset)
