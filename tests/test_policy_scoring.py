from contextlib import nullcontext
from types import SimpleNamespace

import torch
import pytest
from torch import nn

from glimmer_ppo.policy import Policy, PolicyConfig


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(12, 4)
        self.decoder = nn.Linear(4, 4)
        self.head = nn.Linear(4, 12)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head

    def disable_adapter(self):
        return nullcontext()

    def generate(self, input_ids, **kwargs):
        result = input_ids
        for token in (4, 11):
            result = torch.cat([result, torch.tensor([[token]])], dim=1)
            for criterion in kwargs.get('stopping_criteria', []):
                assert criterion(result, None) is False
        return result

    def forward(self, input_ids, attention_mask, logits_to_keep, output_hidden_states, use_cache):
        assert not use_cache
        h = self.decoder(self.embed(input_ids)).tanh()
        return SimpleNamespace(logits=self.head(h[:, -logits_to_keep:]),
                               hidden_states=(h,) if output_hidden_states else None)


@pytest.mark.parametrize('offload', [False, True])
def test_low_memory_score_preserves_values_logprobs_and_gradients(offload):
    torch.manual_seed(0)
    policy = Policy.__new__(Policy)
    nn.Module.__init__(policy)
    policy.model = TinyModel()
    policy.value_head = nn.Linear(4, 1)
    policy.cfg = PolicyConfig(activation_offload=offload)
    record = dict(inputs={'input_ids': torch.tensor([[1, 2, 3]])}, response=torch.tensor([4, 5]))
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    old = policy.model(ids, torch.ones_like(ids), logits_to_keep=3,
                       output_hidden_states=True, use_cache=False)
    expected_logp = old.logits[0, :-1].log_softmax(-1).gather(-1, record['response'][:, None]).squeeze(-1)
    expected_value = policy.value_head(old.hidden_states[-1][0, 2]).squeeze(-1)
    (expected_logp.sum() + expected_value).backward()
    gradients = [p.grad.clone() for p in policy.parameters()]
    policy.zero_grad(set_to_none=True)
    logp, value = policy.score(record)
    torch.testing.assert_close(logp, expected_logp)
    torch.testing.assert_close(value, expected_value)
    (logp.sum() + value).backward()
    for p, expected in zip(policy.parameters(), gradients):
        torch.testing.assert_close(p.grad, expected)
    assert not policy.model.head._forward_pre_hooks
    with torch.no_grad():
        ref_logp, ref_value = policy.score(record, with_value=False, reference=True)
    torch.testing.assert_close(ref_logp, logp)
    assert ref_value is None


def test_prompt_budget_keeps_task_and_complete_recent_turns():
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            return {'input_ids': torch.zeros((1, len(messages) * 10), dtype=torch.long)}
    policy = Policy.__new__(Policy)
    nn.Module.__init__(policy)
    policy.cfg = PolicyConfig(max_prompt_tokens=80)
    policy.processor = Processor()
    messages = [dict(role='system', content='system'), dict(role='user', content='initial task')]
    for i in range(4):
        messages.extend([dict(role='assistant', content=str(i)), dict(role='tool', content=str(i)),
                         dict(role='user', content=f'image {i}')])
    inputs = policy.encode(messages)
    assert inputs['input_ids'].shape[1] == 80
    assert policy.processor.messages == messages[:2] + messages[8:]
    assert len(messages) == 14
    policy.cfg.max_prompt_tokens = 30
    with pytest.raises(RuntimeError, match='two recent turns'):
        policy.encode(messages)


@pytest.mark.parametrize('progress', [False, True])
def test_generation_progress_preserves_sampled_tokens(progress):
    class Processor:
        tokenizer = SimpleNamespace(eos_token_id=11, decode=lambda *a, **k: 'reply')

        def apply_chat_template(self, messages, **kwargs):
            return {'input_ids': torch.tensor([[1, 2, 3]])}
    policy = Policy.__new__(Policy)
    nn.Module.__init__(policy)
    policy.model = TinyModel()
    policy.processor = Processor()
    policy.cfg = PolicyConfig(token_progress=progress, max_new_tokens=8)
    text, truncated, record = policy.act([])
    assert text == 'reply'
    assert not truncated
    assert record['response'].tolist() == [4, 11]


def test_adapter_resume_restores_trainable_parameters_in_place(tmp_path):
    from peft import LoraConfig, get_peft_model
    policy = Policy.__new__(Policy)
    nn.Module.__init__(policy)
    policy.model = get_peft_model(TinyModel(), LoraConfig(r=2, target_modules=['decoder']))
    policy.value_head = nn.Linear(4, 1)
    params = policy.trainable_parameters()
    for p in params:
        p.data.fill_(.25)
    expected = [p.detach().clone() for p in params]
    policy.save(tmp_path)
    for p in params:
        p.data.zero_()
    policy.load(tmp_path)
    assert [id(p) for p in policy.trainable_parameters()] == [id(p) for p in params]
    assert all(p.requires_grad for p in params)
    for p, original in zip(params, expected):
        torch.testing.assert_close(p, original)
