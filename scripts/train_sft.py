"""Imitate verified teacher tool calls with response-only QLoRA cross entropy.

Use --check-only first: it checks multimodal token boundaries without loading
the 30B model. This trainer does not run the simulator or PPO.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

from PIL import Image
import torch
from tqdm.auto import tqdm

from glimmer_ppo.checkpoint import checkpoint_config
from glimmer_ppo.policy import Policy, PolicyConfig
from glimmer_ppo.toolcall import parse_tool_call


def load_examples(dataset):
    rows = [json.loads(line) for line in (dataset / 'examples.jsonl').read_text().splitlines() if line.strip()]
    train = [r for r in rows if r['split'] == 'train']
    validation = [r for r in rows if r['split'] == 'validation']
    train_states = {(r['source']['task_id'], r['source']['init_state']) for r in train}
    validation_states = {(r['source']['task_id'], r['source']['init_state']) for r in validation}
    if not train or not validation or train_states & validation_states:
        raise ValueError('Need nonempty training and validation with disjoint initial states')
    return train, validation


def hydrate(row, dataset):
    messages = copy.deepcopy(row['messages'])
    root = dataset.resolve()
    for message in messages:
        if not isinstance(message.get('content'), list):
            continue
        for part in message['content']:
            if part.get('type') != 'image':
                continue
            path = (root / part['image']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Image path leaves the dataset')
            with Image.open(path) as image:
                part['image'] = image.convert('RGB')
    return messages


def record(policy, row, dataset):
    rec = policy.supervised_record(hydrate(row, dataset), row['target'], row['tools'])
    calls = row['target'].get('tool_calls', [])
    if calls:
        if len(calls) != 1:
            raise ValueError('Expected exactly one target tool call')
        expected = calls[0]['function']
        text = policy.processor.tokenizer.decode(rec['response'], skip_special_tokens=False)
        if parse_tool_call(text) != (expected['name'], expected['arguments']):
            raise ValueError('Native target does not parse back into the original teacher action')
    return rec


def validate(policy, rows, dataset):
    policy.eval()
    loss_sum, tokens = 0., 0
    with torch.no_grad():
        for row in tqdm(rows, desc='SFT validation', unit='example', leave=False):
            rec = record(policy, row, dataset)
            logp, _ = policy.score(rec, with_value=False)
            loss_sum -= float(logp.sum())
            tokens += logp.numel()
    return loss_sum / tokens


def save_weights(policy, destination, config):
    staging = destination.with_name('.' + destination.name + '-' + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    policy.save(staging)
    (staging / 'training_config.json').write_text(json.dumps(config, indent=2) + '\n')
    staging.rename(destination)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--init-policy', type=Path, help='optional adapter weights; optimizer starts fresh')
    p.add_argument('--model', default=PolicyConfig.model)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--accumulate', type=int, default=8)
    p.add_argument('--max-train-examples', type=int)
    p.add_argument('--max-validation-examples', type=int, default=32)
    p.add_argument('--max-prompt-tokens', type=int, default=8192)
    p.add_argument('--activation-offload', action='store_true')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--offline', action='store_true', help='use the cached processor only in --check-only')
    args = p.parse_args()
    if min(args.epochs, args.accumulate, args.max_validation_examples, args.max_prompt_tokens) < 1:
        p.error('Epochs, accumulation, validation count and context limit must be positive')
    if not math.isfinite(args.lr) or args.lr <= 0 or (args.max_train_examples is not None and args.max_train_examples < 1):
        p.error('Learning rate and optional training count must be positive')
    if not args.check_only and args.output is None:
        p.error('Training requires --output')
    if args.output and args.output.exists() and any(f.name != 'train.log' for f in args.output.iterdir()):
        p.error('Use a fresh output directory')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train, validation = load_examples(args.dataset)
    random.shuffle(train)
    random.shuffle(validation)
    train = train[:args.max_train_examples]
    validation = validation[:args.max_validation_examples]
    cfg = PolicyConfig(model=args.model, load_in_4bit=True, temperature=1.,
                       max_prompt_tokens=args.max_prompt_tokens, activation_offload=args.activation_offload)
    if args.init_policy:
        saved = checkpoint_config(args.init_policy)['policy']
        cfg = PolicyConfig(**saved)
        cfg.load_in_4bit = True
        cfg.temperature = 1.
        cfg.gradient_checkpointing = True
        cfg.max_prompt_tokens = args.max_prompt_tokens
        cfg.activation_offload = args.activation_offload
    if args.check_only:
        from transformers import AutoProcessor
        policy = Policy.__new__(Policy)
        torch.nn.Module.__init__(policy)
        policy.cfg = cfg
        policy.processor = AutoProcessor.from_pretrained(cfg.model, local_files_only=args.offline)
        lengths = []
        for row in tqdm(train + validation, desc='Check multimodal examples', unit='example'):
            rec = record(policy, row, args.dataset)
            lengths.append((rec['inputs']['input_ids'].shape[1], rec['response'].numel()))
        print(json.dumps(dict(checked=len(lengths), train=len(train), validation=len(validation),
                              max_prompt_tokens=max(n for n, _ in lengths),
                              max_response_tokens=max(n for _, n in lengths)), indent=2))
        return

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    config = dict(mode='sft', args={k: str(v) for k, v in vars(args).items()}, policy=asdict(cfg),
                  train_examples=len(train), validation_examples=len(validation),
                  note='Weights only; initialize PPO with --init-policy, not --resume. '
                       'Value head is frozen and has not been fitted to this supervised policy.')
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    progress_path = out / 'progress.json'
    (out / 'train.pid').write_text(str(os.getpid()) + '\n')
    print(f'[sft] loading {cfg.model}; train={len(train)} validation={len(validation)}', flush=True)
    policy = Policy(cfg)
    if args.init_policy:
        policy.load(args.init_policy)
    policy.value_head.requires_grad_(False)
    params = [v for v in policy.model.parameters() if v.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    baseline = validate(policy, validation, args.dataset)
    print(f'[sft] initial validation token NLL={baseline:.5f}', flush=True)
    metrics = out / 'metrics.jsonl'
    metrics.write_text(json.dumps(dict(epoch=0, validation_nll=baseline)) + '\n')
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train)
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        bar = tqdm(train, desc=f'SFT epoch {epoch}/{args.epochs}', unit='example')
        for index, row in enumerate(bar):
            rec = record(policy, row, args.dataset)
            logp, _ = policy.score(rec, with_value=False)
            loss = -logp.mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite supervised loss')
            group_start = (index // args.accumulate) * args.accumulate
            group_size = min(args.accumulate, len(train) - group_start)
            (loss / group_size).backward()
            if (index + 1) % args.accumulate == 0 or index + 1 == len(train):
                torch.nn.utils.clip_grad_norm_(params, 1.)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            bar.set_postfix(loss=f'{losses[-1]:.4f}')
            progress_path.write_text(json.dumps(dict(phase='training', epoch=epoch,
                completed=index + 1, total=len(train), loss=losses[-1])) + '\n')
            tqdm.write(f'[sft] epoch={epoch} example={index + 1}/{len(train)} '
                       f'call={row["call"]} loss={losses[-1]:.5f}')
            del rec, logp, loss
        val = validate(policy, validation, args.dataset)
        summary = dict(epoch=epoch, training_nll=sum(losses) / len(losses), validation_nll=val,
                       seconds=time.time() - started)
        with metrics.open('a') as stream:
            stream.write(json.dumps(summary) + '\n')
        destination = out / f'epoch_{epoch:04d}'
        save_weights(policy, destination, config)
        print(f'[sft] saved {destination}; validation token NLL={val:.5f}', flush=True)
    progress_path.write_text(json.dumps(dict(phase='complete', epochs=args.epochs)) + '\n')
    print('[sft] complete; evaluate robot success before further PPO', flush=True)


if __name__ == '__main__':
    main()
