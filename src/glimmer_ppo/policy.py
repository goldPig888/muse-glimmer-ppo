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
    activation_offload: bool = False
    max_prompt_tokens: int | None = None
    token_progress: bool = False
    template_kwargs: dict = field(default_factory=dict)


def _load_base(cfg: PolicyConfig):
    import transformers
    dtype = getattr(torch, cfg.dtype)
    kwargs = dict(torch_dtype=dtype, device_map='auto')
    if cfg.load_in_4bit:
        kwargs['quantization_config'] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=dtype,
            # Muse casts image activations to patch_embedding.weight.dtype.
            # A packed 4-bit weight is uint8, which breaks the next LayerNorm.
            llm_int8_skip_modules=['lm_head', 'patch_embedding'])
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
        # Not peft.prepare_model_for_kbit_training: it upcasts every bf16 tensor to fp32,
        # which for Muse means the 202k-row embeddings, output head and vision tower
        # (about 18 GB more) and would not fit a 48 GB A6000.
        if cfg.gradient_checkpointing:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
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
    def encode(self, messages, tools=None) -> dict:
        return self._prepare_messages(messages, tools)[1]

    def _prepare_messages(self, messages, tools=None):
        kwargs = dict(self.cfg.template_kwargs)
        if tools:
            kwargs['tools'] = tools
        if self.cfg.reasoning_strength:
            kwargs['reasoning_strength'] = self.cfg.reasoning_strength
        messages = list(messages)
        while True:
            inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                         return_dict=True, return_tensors='pt', **kwargs)
            limit = self.cfg.max_prompt_tokens
            if limit is None or inputs['input_ids'].shape[1] <= limit:
                return messages, inputs
            # Keep the system and initial user task, plus at least two recent
            # complete turns. Never cut tokenized image blocks or tool calls.
            starts = [i for i, m in enumerate(messages) if m['role'] == 'assistant']
            if len(starts) <= 2:
                raise RuntimeError(f'Prompt exceeds {limit} tokens even with only two recent turns; '
                                   'increase --max-prompt-tokens or reduce image resolution')
            messages = messages[:starts[0]] + messages[starts[1]:]

    def _device(self):
        return self.model.get_input_embeddings().weight.device

    def _compute_context(self):
        # PEFT stores adapters in fp32. Autocast keeps their large projection
        # activations in bf16, matching the base model's compute dtype.
        return (torch.autocast('cuda', dtype=torch.bfloat16)
                if self._device().type == 'cuda' and self.cfg.dtype == 'bfloat16'
                else nullcontext())

    # -- acting ------------------------------------------------------------
    @torch.no_grad()
    def act(self, messages, tools=None):
        """Sample one reply. Returns (text, truncated, record) where record holds
        everything PPO needs to rescore this decision later (on CPU)."""
        inputs = self.encode(messages, tools)
        dev_inputs = {k: v.to(self._device()) if torch.is_tensor(v) else v for k, v in inputs.items()}
        prompt_len = inputs['input_ids'].shape[1]
        generation_kwargs = {}
        if self.cfg.token_progress:
            from transformers import StoppingCriteria, StoppingCriteriaList
            from tqdm.auto import tqdm
            bar = tqdm(total=self.cfg.max_new_tokens, desc='Generate tokens', unit='tok',
                       position=2, leave=False, mininterval=1.0, dynamic_ncols=True)

            class Progress(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    bar.update(input_ids.shape[1] - prompt_len - bar.n)
                    return False

            generation_kwargs['stopping_criteria'] = StoppingCriteriaList([Progress()])
        else:
            bar = None
        try:
            with self._compute_context():
                out = self.model.generate(**dev_inputs, max_new_tokens=self.cfg.max_new_tokens, do_sample=True,
                                          max_length=None, temperature=self.cfg.temperature, top_p=1.0, top_k=0,
                                          **generation_kwargs)
        finally:
            if bar is not None:
                bar.close()
        response = out[0, prompt_len:].cpu()
        eos = self.processor.tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        truncated = len(response) >= self.cfg.max_new_tokens and int(response[-1]) not in eos_ids
        # Keep special tokens: tool-call markers are often special tokens.
        text = self.processor.tokenizer.decode(response, skip_special_tokens=False)
        record = dict(inputs={k: v.cpu() if torch.is_tensor(v) else v for k, v in inputs.items()},
                      response=response)
        return text, truncated, record

    # -- external rollout engine (llama.cpp) ----------------------------------
    @property
    def image_token(self) -> str:
        """The placeholder the chat template writes for each image."""
        return getattr(self.processor, 'image_token', None) or '<|image|>'

    def render_prompt(self, messages, tools=None):
        """The exact prompt text this model would see, plus its images in order.

        Used to drive llama.cpp with the same prompt Hugging Face would build."""
        if self.cfg.max_prompt_tokens is not None:
            messages, _ = self._prepare_messages(messages, tools)
        kwargs = dict(self.cfg.template_kwargs)
        if tools:
            kwargs['tools'] = tools
        if self.cfg.reasoning_strength:
            kwargs['reasoning_strength'] = self.cfg.reasoning_strength
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kwargs)
        images = [part['image'] for m in messages if isinstance(m.get('content'), list)
                  for part in m['content'] if part.get('type') == 'image']
        return text, images

    def record_for(self, messages, tools, response_ids) -> dict:
        """A PPO record for a reply sampled elsewhere: HF-encoded prompt + the sampled token ids."""
        inputs = self.encode(messages, tools)
        return dict(inputs={k: v.cpu() if torch.is_tensor(v) else v for k, v in inputs.items()},
                    response=torch.as_tensor(response_ids, dtype=torch.long))

    def save_adapter(self, path):
        """LoRA adapter only (what llama.cpp needs)."""
        self.model.save_pretrained(path)

    # -- scoring -----------------------------------------------------------
    def _forward_with_value(self, with_value, **inputs):
        """Capture only the first requested final-layer token for the critic.

        Callers slice logits to start at the last prompt token. Reading the
        output head's input gives that token's final hidden state without
        retaining every decoder layer's hidden states for the full context.
        Scoring never needs an autoregressive KV cache.
        """
        hidden = []
        handle = None
        if with_value:
            def capture(_module, args):
                hidden.append(args[0][0, 0].clone())
            handle = self.model.get_output_embeddings().register_forward_pre_hook(capture)
        try:
            with self._compute_context():
                out = self.model(**inputs, output_hidden_states=False, use_cache=False)
        finally:
            if handle is not None:
                handle.remove()
        if with_value and not hidden:
            raise RuntimeError('Output head did not expose the critic hidden state')
        value = None
        if with_value:
            h = hidden[0].float().to(self.value_head.weight.device)
            value = self.value_head(h).squeeze(-1)
        return out, value

    def score(self, record, with_value=True, reference=False):
        """Per-token log-probs of the stored response, and V(s) at the last prompt token."""
        ctx = (torch.autograd.graph.save_on_cpu(pin_memory=True)
               if self.cfg.activation_offload and torch.is_grad_enabled() else nullcontext())
        with ctx:
            return self._score(record, with_value, reference)

    def _score(self, record, with_value, reference):
        inputs = {k: v.to(self._device()) if torch.is_tensor(v) else v for k, v in record['inputs'].items()}
        response = record['response'].to(self._device())
        prompt_len = inputs['input_ids'].shape[1]
        ids = torch.cat([inputs['input_ids'], response[None]], dim=1)
        mask = torch.ones_like(ids)
        rest = {k: v for k, v in inputs.items() if k not in ('input_ids', 'attention_mask')}
        ctx = self.model.disable_adapter() if reference else nullcontext()
        n = response.shape[0]
        with ctx:
            # logits_to_keep: only the positions that predict the response (vocab is 202k).
            out, value = self._forward_with_value(with_value, input_ids=ids, attention_mask=mask,
                                                   logits_to_keep=n + 1, **rest)
        logits = out.logits[0, :-1].float() / self.cfg.temperature
        logp = torch.log_softmax(logits, dim=-1).gather(-1, response[:, None]).squeeze(-1)
        return logp, value

    @torch.no_grad()
    def value_of(self, messages, tools=None) -> float:
        """V(s) for a state with no response yet (used to bootstrap a batch cut mid-episode)."""
        inputs = {k: v.to(self._device()) if torch.is_tensor(v) else v for k, v in self.encode(messages, tools).items()}
        _, value = self._forward_with_value(True, **inputs, logits_to_keep=1)
        return float(value)

    def save(self, path):
        self.model.save_pretrained(path)
        torch.save(self.value_head.state_dict(), f'{path}/value_head.pt')
