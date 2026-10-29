import pytest
import torch.nn as nn

from glimmer_ppo.policy import PolicyConfig, lora_target_names


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(4, 4) for _ in range(4))
        self.gate_proj, self.up_proj, self.down_proj = (nn.Linear(4, 4) for _ in range(3))


class FakeMuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = nn.ModuleList([Block()])
        self.multi_modal_projector = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        self.language_model = nn.ModuleDict(dict(layers=nn.ModuleList([Block(), Block()])))
        self.lm_head = nn.Linear(4, 10)


def test_targets_only_text_decoder_linears():
    names = lora_target_names(FakeMuse(), set(PolicyConfig.lora_targets))
    assert len(names) == 14                       # 2 layers x 7 projections
    assert all(n.startswith('language_model.') for n in names)


def test_no_match_is_an_error():
    with pytest.raises(RuntimeError):
        lora_target_names(FakeMuse(), {'fc1'})
