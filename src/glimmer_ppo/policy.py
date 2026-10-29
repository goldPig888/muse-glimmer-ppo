"""Muse Glimmer as actor and critic: LoRA on the language model, a value head.

* Actor: the base model with LoRA adapters on the text decoder's linear layers.
  The vision tower, projector, embeddings and output head stay frozen.
* Critic: one linear layer on the last hidden state at the final prompt token,
  i.e. V(s_t) for the decision about to be made.
* Reference: the same model with the adapter disabled (no second copy).

Log-probs used by PPO always come from ``score`` (this model, this precision),
never from the sampling engine, so the PPO ratio is consistent.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
import torch.nn as nn

# Linear layers whose name contains one of these are never adapted.
FROZEN_PARTS = ('vision', 'visual', 'patch', 'projector', 'multi_modal', 'mm_', 'lm_head', 'embed')


@dataclass
class PolicyConfig:
    model: str = 'meta-models/Muse-Glimmer-30B'
    dtype: str = 'bfloat16'
    load_in_4bit: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    # Leaf names of linear layers to adapt; matched only outside FROZEN_PARTS.
    lora_targets: tuple = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
    reasoning_strength: str | None = 'high'
    max_new_tokens: int = 2048
    temperature: float = 1.0
    gradient_checkpointing: bool = True
    template_kwargs: dict = field(default_factory=dict)


def _load_base(cfg: PolicyConfig):
    import transformers
    dtype = getattr(torch, cfg.dtype)
    kwargs = dict(torch_dtype=dtype, device_map='auto')
    if cfg.load_in_4bit:
        kwargs['quantization_config'] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=dtype)
    errors = []
    for name in ('AutoModelForMultimodalLM', 'AutoModelForImageTextToText', 'MuseGlimmerForConditionalGeneration'):
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            return cls.from_pretrained(cfg.model, **kwargs)
        except (ValueError, KeyError) as exc:
            errors.append(f'{name}: {exc}')
    raise RuntimeError('Could not load the model; upgrade transformers.\n' + '\n'.join(errors))


def lora_target_names(model, leaves) -> list[str]:
    """Full module names of text-decoder linear layers to adapt."""
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(part in name.lower() for part in FROZEN_PARTS):
            continue
        if name.rsplit('.', 1)[-1] in leaves:
            names.append(name)
    if not names:
        raise RuntimeError('No LoRA targets matched. Run scripts/print_modules.py and set lora_targets.')
    return names


class Policy(nn.Module):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor
        self.cfg = cfg
        self.processor = AutoProcessor.from_pretrained(cfg.model)
        base = _load_base(cfg)
        for p in base.parameters():
            p.requires_grad_(False)
        if cfg.load_in_4bit:
            from peft import prepare_model_for_kbit_training
            base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=cfg.gradient_checkpointing)
        elif cfg.gradient_checkpointing:
            base.gradient_checkpointing_enable()
            base.enable_input_require_grads()
        targets = lora_target_names(base, set(cfg.lora_targets))
        self.model = get_peft_model(base, LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                                                     lora_dropout=cfg.lora_dropout, target_modules=targets,
                                                     bias='none'))
        hidden = self._hidden_size(base)
        self.value_head = nn.Linear(hidden, 1, dtype=torch.float32)
        nn.init.normal_(self.value_head.weight, std=1e-3)
        nn.init.zeros_(self.value_head.bias)
        self.value_head.to(self.model.get_output_embeddings().weight.device)
        self.n_lora_targets = len(targets)

    @staticmethod
    def _hidden_size(model) -> int:
        c = model.config
        for owner in (getattr(c, 'text_config', None), c):
            if owner is not None and getattr(owner, 'hidden_size', None):
                return int(owner.hidden_size)
        raise RuntimeError('hidden_size not found in config')

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad] + list(self.value_head.parameters())

    # -- inputs ------------------------------------------------------------
    def encode(self, messages) -> dict:
        kwargs = dict(self.cfg.template_kwargs)
        if self.cfg.reasoning_strength:
            kwargs['reasoning_strength'] = self.cfg.reasoning_strength
        return self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                  return_dict=True, return_tensors='pt', **kwargs)

    def _device(self):
        return self.model.get_input_embeddings().weight.device

    # -- acting ------------------------------------------------------------
    @torch.no_grad()
    def act(self, messages):
        """Sample one reply. Returns (text, truncated, record) where record holds
        everything PPO needs to rescore this decision later (on CPU)."""
        inputs = self.encode(messages)
        dev_inputs = {k: v.to(self._device()) if torch.is_tensor(v) else v for k, v in inputs.items()}
        out = self.model.generate(**dev_inputs, max_new_tokens=self.cfg.max_new_tokens, do_sample=True,
                                  temperature=self.cfg.temperature, top_p=1.0, top_k=0)
        prompt_len = inputs['input_ids'].shape[1]
        response = out[0, prompt_len:].cpu()
        eos = self.processor.tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        truncated = len(response) >= self.cfg.max_new_tokens and int(response[-1]) not in eos_ids
        text = self.processor.tokenizer.decode(response, skip_special_tokens=True)
        record = dict(inputs={k: v.cpu() if torch.is_tensor(v) else v for k, v in inputs.items()},
                      response=response)
        return text, truncated, record

    # -- scoring -----------------------------------------------------------
    def score(self, record, with_value=True, reference=False):
        """Per-token log-probs of the stored response, and V(s) at the last prompt token."""
        inputs = {k: v.to(self._device()) if torch.is_tensor(v) else v for k, v in record['inputs'].items()}
        response = record['response'].to(self._device())
        prompt_len = inputs['input_ids'].shape[1]
        ids = torch.cat([inputs['input_ids'], response[None]], dim=1)
        mask = torch.ones_like(ids)
        rest = {k: v for k, v in inputs.items() if k not in ('input_ids', 'attention_mask')}
        ctx = self.model.disable_adapter() if reference else nullcontext()
        with ctx:
            out = self.model(input_ids=ids, attention_mask=mask, output_hidden_states=with_value, **rest)
        logits = out.logits[0, prompt_len - 1:-1].float() / self.cfg.temperature
        logp = torch.log_softmax(logits, dim=-1).gather(-1, response[:, None]).squeeze(-1)
        value = None
        if with_value:
            h = out.hidden_states[-1][0, prompt_len - 1].float().to(self.value_head.weight.device)
            value = self.value_head(h).squeeze(-1)
        return logp, value

    @torch.no_grad()
    def value_of(self, messages) -> float:
        """V(s) for a state with no response yet (used to bootstrap a batch cut mid-episode)."""
        inputs = {k: v.to(self._device()) if torch.is_tensor(v) else v for k, v in self.encode(messages).items()}
        out = self.model(**inputs, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1].float().to(self.value_head.weight.device)
        return float(self.value_head(h))

    def save(self, path):
        self.model.save_pretrained(path)
        torch.save(self.value_head.state_dict(), f'{path}/value_head.pt')
