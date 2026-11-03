import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from glimmer_ppo.captain_rl import RequestReservoir, actor_loss, grouped_credit, join_requests


def trainer_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts'))
    spec = importlib.util.spec_from_file_location('captain_trainer_test', Path(__file__).parents[1] / 'scripts/train_captain_rl.py')
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return trainer


def test_request_identity_and_uniform_subsampling_include_all_metadata():
    reservoir = RequestReservoir(3, 123)
    driver = []
    for i in range(20):
        rid = f'response-{i}'
        if reservoir.wants(rid, 2):
            reservoir.add(rid, {'response': torch.tensor([i])})
        driver.append(dict(index=i + 1, response_id=rid))
    assert len(reservoir.metadata) == 20 and len(reservoir.records) == 3
    assert join_requests(driver, reservoir.metadata, reservoir.records)['response-0'] == 1
    with pytest.raises(ValueError, match='IDs do not match'):
        join_requests(driver[::-1], reservoir.metadata, reservoir.records)


def test_group_outcome_and_same_episode_centering_include_text_only_failures():
    credit = grouped_credit([{'a': .3, 'b': .5}, None], [1., 0.], [['a','b','final'], ['bad']], 1.)
    assert credit[0] == pytest.approx({'a':.4, 'b':.6, 'final':.5})
    assert credit[1] == {'bad':-.5}


def test_actor_loss_gradient_direction_and_ratio_mask():
    value = torch.tensor([-.7], requires_grad=True)
    loss, _ = actor_loss(value, value.detach(), value.detach(), 1.)
    loss.backward()
    assert value.grad.item() < 0  # Increase the probability of rewarded sampled tokens.
    value = torch.tensor([-.7], requires_grad=True)
    loss, _ = actor_loss(value, value.detach(), value.detach(), 1., pure_error=True, code_cost=-1.)
    loss.backward()
    assert value.grad.item() > 0  # Pure failures cannot receive positive task credit.
    value = torch.tensor([0.], requires_grad=True)
    loss, stats = actor_loss(value, torch.tensor([-3.]), value.detach(), 1.)
    loss.backward()
    assert stats['kept_fraction'] == 0 and value.grad.item() == 0


def test_inclusion_weighting_matches_full_request_objective():
    losses = []
    for logp in (-.2, -.4):
        value = torch.tensor([logp])
        loss, _ = actor_loss(value, value, value, 1., trajectory_tokens=2, inclusion=1.)
        losses.append(loss)
    estimated = []
    for logp in (-.2, -.4):
        value = torch.tensor([logp])
        loss, _ = actor_loss(value, value, value, 1., trajectory_tokens=2, inclusion=2.)
        estimated.append(loss)
    assert torch.allclose(sum(losses), sum(estimated) / 2)


def test_training_capture_is_usable_by_autograd_after_generation(tmp_path, monkeypatch):
    from glimmer_ppo.captain_bridge import PYTHON_TOOL
    trainer = trainer_module(monkeypatch)
    raw = (' to=python<|message|><atem:function_calls><atem:invoke name="python">'
           '<atem:parameter name="input">print(41)</atem:parameter>'
           '</atem:invoke></atem:function_calls><|eot|>')

    class Processor:
        tokenizer = SimpleNamespace(decode=lambda *a, **kw: raw, eos_token_id=9)

        def apply_chat_template(self, *a, **kw):
            return dict(input_ids=torch.tensor([[1, 2]]), pixel_values=torch.tensor([[.3]]))

    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[9]),
                            generate=lambda **kw: torch.tensor([[1, 2, 8, 9]]))
    policy = SimpleNamespace(model=model, processor=Processor(), _device=lambda: torch.device('cpu'))
    args = SimpleNamespace(model='fixture', served_name='glimmer-rl', output=tmp_path,
        rounds=1, max_training_requests=8, max_new_tokens=4, context_tokens=64, reasoning='high')
    backend = trainer.TrainingBackend(args, policy)
    backend.begin_episode(123)
    response, _ = backend.generate(dict(model='glimmer-rl', tools=[PYTHON_TOOL],
        messages=[dict(role='user', content='test')], max_tokens=4))
    capture = backend.end_episode()
    record = capture.records[response['id']]
    assert not record['response'].is_inference()
    assert not record['inputs']['pixel_values'].is_inference()
    weight = torch.tensor(2., requires_grad=True)
    (weight * record['inputs']['pixel_values']).sum().backward()
    assert weight.grad.item() == pytest.approx(.3)


def test_launcher_tee_directory_is_allowed_and_startup_errors_are_saved(tmp_path, monkeypatch):
    trainer = trainer_module(monkeypatch)
    (tmp_path / 'train.log').write_text('launcher log\n')
    monkeypatch.setattr(trainer, 'load_upstream', lambda path: (None, None, None, None))

    class Socket:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def bind(self, *a): pass

    monkeypatch.setattr(trainer.socket, 'socket', Socket)
    def unavailable(*a, **kw):
        raise RuntimeError('fixture service unavailable')
    monkeypatch.setattr(trainer, 'urlopen', unavailable)
    monkeypatch.setattr(sys, 'argv', ['train_captain_rl.py', '--output', str(tmp_path),
        '--harness-python', sys.executable, '--student-python', sys.executable])
    with pytest.raises(RuntimeError, match='fixture service unavailable'):
        trainer.main()
    assert json.loads((tmp_path / 'config.json').read_text())['mode'] == 'captain_rl'
    assert json.loads((tmp_path / 'progress.json').read_text())['phase'] == 'error'


def test_full_cpu_update_with_authors_critic_and_features(tmp_path, monkeypatch):
    """Optional real upstream integration; no simulator, model download or GPU needed."""
    checkout, critic_dir = os.getenv('CAPTAIN_TEST_CHECKOUT'), os.getenv('CAPTAIN_TEST_CRITIC')
    if not checkout or not critic_dir:
        pytest.skip('set CAPTAIN_TEST_CHECKOUT and CAPTAIN_TEST_CRITIC for upstream integration')
    trainer = trainer_module(monkeypatch)
    pc, cc, _, _ = trainer.load_upstream(Path(checkout))
    critic = cc.CoCritic(critic_dir, hidden_dim=0, use_hidden=False).eval()
    class TinyActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([.1, -.1]))

        def score(self, record, with_value=False, reference=False):
            weights = self.weight.detach() if reference else self.weight
            return torch.log_softmax(weights, 0)[record['response']], None

        def trainable_parameters(self):
            return list(self.parameters())

        def save(self, path):
            torch.save(self.state_dict(), str(Path(path) / 'actor.pt'))

    actor = TinyActor()
    initial_actor = actor.weight.detach().clone()
    initial_critic = [p.detach().clone() for p in critic.parameters()]
    trajectories=[]
    for i in range(4):
        folder=tmp_path / f'episode{i}'
        (folder/'driver').mkdir(parents=True)
        driver, marks = [], []
        capture=RequestReservoir(8,123+i)
        for request in (1,2):
            rid=f'ep{i}-r{request}'
            driver.append(dict(index=request,response_id=rid,replayed=False))
            marks.append(dict(request_index=request,call_index=0,eef=[.45,.1,.2],objects={},
                goal_states={'goal_1':dict(satisfied=False,target_position_m=[.5,.1,.1],destination_position_m=[.6,.1,.1],distance_to_goal=.1)}))
            capture.wants(rid,1)
            capture.metadata[-1]['parse_errors']=False
            capture.add(rid,dict(response=torch.tensor([0 if i<2 else 1])))
        for name,rows in [('driver/requests.jsonl',driver),('hidden_request_marks.jsonl',marks)]:
            (folder/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        score=dict(suite='libero_10',task_id=0,episode_dir=str(folder),sample_path='fixture',
                   verdict='hidden_success' if i<2 else 'acted',diagnostics=dict(host_final_success=i<2,
                   termination_reason='agent_finished' if i<2 else 'request_cap_exhausted'))
        trajectories.append(dict(score=score,capture=capture,labels={},indices={r['response_id']:r['index'] for r in driver}))
    actor_opt=torch.optim.AdamW(actor.parameters(),lr=.01)
    critic_opt=torch.optim.AdamW(critic.parameters(),lr=1e-4)
    args=SimpleNamespace(max_requests=60,kl_beta=.002,model='tiny')
    sigma,metrics=trainer.optimize_group(actor,actor_opt,critic,critic_opt,cc,pc,trajectories,
        dict(missed_finish_return=.25,exhaust_reward=-.1),None,args,lambda **kw:None)
    assert not torch.equal(actor.weight,initial_actor)
    assert any(not torch.equal(a,b) for a,b in zip(initial_critic,critic.parameters()))
    assert metrics['trained_requests']==8 and metrics['critic_bce']>0 and sigma>0
    path=tmp_path/'checkpoint'
    trainer.checkpoint(path,actor,actor_opt,critic,critic_opt,1,sigma,args)
    saved=torch.load(path/'trainer.pt',map_location='cpu',weights_only=False)
    assert saved['round']==1 and saved['actor_optimizer']['state']
    restored_actor=TinyActor()
    restored_actor.load_state_dict(torch.load(path/'actor.pt',weights_only=True))
    restored_actor_opt=torch.optim.AdamW(restored_actor.parameters(),lr=.01)
    restored_actor_opt.load_state_dict(saved['actor_optimizer'])
    restored_critic=cc.CoCritic(critic_dir,hidden_dim=0,use_hidden=False).eval()
    restored_critic.load_state_dict(saved['critic'])
    restored_critic_opt=torch.optim.AdamW(restored_critic.parameters(),lr=1e-4)
    restored_critic_opt.load_state_dict(saved['critic_optimizer'])
    for model,optimizer,value,value_optimizer in ((actor,actor_opt,critic,critic_opt),
        (restored_actor,restored_actor_opt,restored_critic,restored_critic_opt)):
        trainer.optimize_group(model,optimizer,value,value_optimizer,cc,pc,trajectories,
            dict(missed_finish_return=.25,exhaust_reward=-.1),sigma,args,lambda **kw:None)
    assert torch.equal(actor.weight,restored_actor.weight)
    assert all(torch.equal(a,b) for a,b in zip(critic.parameters(),restored_critic.parameters()))
