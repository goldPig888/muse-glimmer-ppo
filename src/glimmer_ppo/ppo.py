"""PPO math, one decision (one program) per time step. No model code here.

Notation follows the plan doc: r_t reward of decision t, V(s_t) critic value,
A_t advantage (shared by every token of decision t's program), R_t value target.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PPOConfig:
    gamma: float = 1.0
    lam: float = 0.95
    clip: float = 0.2
    value_clip: float | None = 0.2
    vf_coef: float = 0.5
    kl_coef: float = 0.05          # beta; KL to the reference is subtracted from the reward
    epochs: int = 2
    decisions_per_update: int = 64
    lr: float = 1e-5
    value_lr: float | None = None  # defaults to lr
    max_grad_norm: float = 1.0
    critic_warmup_updates: int = 10
    normalize_advantages: bool = True


def gae(rewards, values, dones, gamma: float, lam: float, last_value: float = 0.0):
    """Generalized advantage estimation over a flat list of decisions.

    ``dones[t]`` is True when decision t ended its episode, so nothing is
    bootstrapped across episodes. ``last_value`` bootstraps a batch that ends
    mid-episode. Returns (advantages, returns), both float tensors.
    """
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    values = torch.as_tensor(values, dtype=torch.float32)
    dones = torch.as_tensor(dones, dtype=torch.float32)
    n = len(rewards)
    adv = torch.zeros(n)
    running = 0.0
    for t in reversed(range(n)):
        next_value = last_value if t == n - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        running = delta + gamma * lam * nonterminal * running
        adv[t] = running
    return adv, adv + values


def kl_penalty(logp_old: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """Sum over a program's tokens of log pi_old - log pi_ref (the 'k1' estimate)."""
    return (logp_old - logp_ref).sum()


def policy_loss(logp: torch.Tensor, logp_old: torch.Tensor, advantage: torch.Tensor, clip: float):
    """Clipped surrogate for one decision; every token shares ``advantage``.

    Returns (loss, stats). Loss is averaged over the program's tokens.
    """
    log_ratio = logp - logp_old
    ratio = log_ratio.exp()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1 - clip, 1 + clip) * advantage
    loss = -torch.min(unclipped, clipped).mean()
    with torch.no_grad():
        stats = dict(clipfrac=((ratio - 1).abs() > clip).float().mean().item(),
                     approx_kl=((ratio - 1) - log_ratio).mean().item())
    return loss, stats


def value_loss(value: torch.Tensor, value_old: torch.Tensor, target: torch.Tensor, clip: float | None):
    loss = (value - target) ** 2
    if clip is not None:
        clipped = value_old + (value - value_old).clamp(-clip, clip)
        loss = torch.max(loss, (clipped - target) ** 2)
    return 0.5 * loss


def explained_variance(values, returns) -> float:
    values = torch.as_tensor(values, dtype=torch.float32)
    returns = torch.as_tensor(returns, dtype=torch.float32)
    var = returns.var(unbiased=False)
    return float('nan') if var == 0 else float(1 - (returns - values).var(unbiased=False) / var)
