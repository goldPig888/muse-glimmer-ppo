"""Glimmer adaptation of CaPtain's single-pass actor update.

The unchanged upstream critic/advantage implementation is imported by the
launcher. These helpers do not implement a replacement reward or PPO objective.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from urllib.request import urlopen

import torch


CRITIC_HASHES = {
    'feature_spec.json': '38ce99755d758b9196bd5e16437f543f63f78386f4c223eaf71348f6b086e8a7',
    'critic_default.pt': '4ee584905a796e0d6a1eb17935d1f9bdf2574b4b21cccee8500945dea925ef99',
}


def ensure_critic(folder):
    """Download only the released 3 MB critic, checking exact published hashes."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in CRITIC_HASHES.items():
        path = folder / name
        if not path.exists():
            url = 'https://huggingface.co/datasets/YijiaFan/CaPtain-artifacts/resolve/main/critic/warm_start/' + name
            with urlopen(url, timeout=120) as response:
                payload = response.read()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f'{name}: released critic hash mismatch')
            staging = path.with_suffix(path.suffix + '.tmp')
            staging.write_bytes(payload)
            staging.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'{path}: expected the released critic; use a fresh directory')
    return folder


class RequestReservoir:
    """Uniform request subsampling, independent of outcome, token length or advantage."""
    def __init__(self, maximum, seed):
        self.maximum = maximum
        self.rng = random.Random(seed)
        self.metadata, self.records = [], {}

    def wants(self, response_id, tokens):
        index = len(self.metadata)
        self.metadata.append(dict(response_id=response_id, tokens=tokens))
        slot = index if index < self.maximum else self.rng.randrange(index + 1)
        if slot >= self.maximum:
            return False
        if index >= self.maximum:
            old = list(self.records)[slot]
            del self.records[old]
        return True

    def add(self, response_id, record):
        self.records[response_id] = record


def join_requests(driver_rows, metadata, records):
    """Fail closed on retries, missing requests, or mismatched response identity."""
    actual = [r['response_id'] for r in driver_rows if not r.get('replayed')]
    recorded = [r['response_id'] for r in metadata]
    if actual != recorded or len(set(actual)) != len(actual) or not actual:
        raise ValueError('rollout request IDs do not match exactly; no training on this episode')
    if not set(records).issubset(actual):
        raise ValueError('subsampled request absent from the executed episode')
    return {r['response_id']: int(r['index']) for r in driver_rows}


def grouped_credit(gae, totals, response_ids, sigma):
    """Group outcome + centered physical GAE; text-only requests have no physical delta."""
    base = sum(totals) / len(totals)
    out = []
    for advantages, total, ids in zip(gae, totals, response_ids):
        advantages = advantages or {}
        mean = sum(advantages.values()) / len(advantages) if advantages else 0.
        out.append({rid: (total - base + (advantages[rid] - mean if rid in advantages else 0.)) / sigma for rid in ids})
    return out


def actor_loss(logp, old, reference, advantage, *, pure_error=False, code_cost=0.0,
               inclusion=1.0, trajectory_tokens=1, trajectories=1, horizon=60,
               kl_beta=.002, code_weight=.1):
    """Detached ratio masking (.2, 4), trajectory-mean weighting, k3 KL.

    Full-distribution HF sampling needs no top-k score-centering correction.
    A pure code failure receives no positive task credit plus the local code cost.
    """
    if pure_error:
        advantage = min(float(advantage), 0.0)
    ratio = torch.exp((logp.detach() - old.to(logp.device)).clamp(-60, 60))
    keep = ((ratio > .2) & (ratio < 4)).to(logp.dtype)
    score = ratio * keep * logp
    difference = reference.to(logp.device) - logp
    kl = torch.expm1(difference) - difference
    # Match upstream RL_KL_ON_KEPT: restore total token mass after masking.
    kept = keep.sum()
    kl_sum = (kl * keep).sum() * (keep.numel() / kept.clamp_min(1))
    scale = inclusion / (trajectory_tokens * trajectories)
    loss = scale * (-float(advantage) * score.sum() + kl_beta * kl_sum)
    loss = loss - inclusion * code_weight * float(code_cost) * score.mean() / (horizon * trajectories)
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite actor objective; optimizer not stepped')
    return loss, dict(policy_loss=float((-float(advantage) * score.mean()).detach()),
                      kl=float((kl_sum / keep.numel()).detach()), kept_fraction=float(keep.mean()))


def atomic_json(path, value):
    path = Path(path)
    staging = path.with_suffix(path.suffix + '.tmp')
    staging.write_text(json.dumps(value, default=str, allow_nan=False, indent=2) + '\n')
    staging.replace(path)


def jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def fresh_seeds(harness, suite, task, seed):
    candidates = []
    for path in (Path(harness) / 'data/libero_fresh_init' / suite).glob('*.json'):
        meta = json.loads(path.read_text())
        if meta.get('task_id') == task:
            candidates.append((path, meta))
    if len(candidates) != 1:
        raise ValueError('cannot identify unique upstream fresh-state bank for task')
    path, meta = candidates[0]
    states = list(meta['reset_seeds'])
    if not states or any(s < 1000 for s in states):
        raise ValueError('training requires upstream fresh seeds >= 1000, not evaluation states')
    data = path.with_suffix('')
    if hashlib.sha256(data.read_bytes()).hexdigest() != meta['sha256']:
        raise ValueError('fresh-state bank hash mismatch')
    random.Random(seed).shuffle(states)
    return states, dict(file=str(data), sha256=meta['sha256'], count=len(states))
