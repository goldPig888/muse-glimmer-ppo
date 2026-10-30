"""Per-decision reward from the simulator. Pure Python so it is unit-tested.

The model's own ``done`` is never rewarded: only LIBERO's predicates are.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

OK = {'completed', 'success'}
NEUTRAL = {'grasp_failed', 'dropped', 'observation', 'observation_limit'}


@dataclass
class RewardConfig:
    success: float = 1.0        # LIBERO success predicate became true
    progress: float = 0.5       # times the change in fraction of goal predicates met
    grasp: float = 0.2          # once per episode: first grasp of an object of interest
    rejected: float = -0.05     # plan_error, IK_FAILURE, COLLISION, blocked, truncated reply, ...
    repeat: float = -0.1        # identical program to an earlier failed one, from the same pose
    per_decision: float = -0.005


def status_of(feedback: str) -> str:
    """Session.feedback is JSON from execute(), or a plain string for errors."""
    try:
        return str(json.loads(feedback).get('status', 'unknown'))
    except (json.JSONDecodeError, AttributeError, TypeError):
        text = (feedback or '').strip()
        if text.startswith('New observation requested'):
            return 'observation'
        if text.startswith('observation_limit'):
            return 'observation_limit'
        if text.startswith('Response truncated'):
            return 'truncated'
        if text.startswith('plan_error'):
            return 'plan_error'
        return text.split(':')[0] or 'unknown'


@dataclass
class DecisionReward:
    total: float
    parts: dict = field(default_factory=dict)


@dataclass
class PotentialConfig:
    """Reward for the paper-protocol env: success + potential-based shaping + small costs.

    r_t = R*1[success] + Phi(s_{t+1}) - Phi(s_t) - c_d - c_inv*1[invalid] - c_rep*1[repeat]
    Phi(s) = w_hold*1[holding target] + w_lift*1[lifted >= 3 cm] + w_goal*(fraction of goal conditions met)
    Set ``shaping=False`` for the sparse-success ablation.
    """
    success: float = 1.0
    w_hold: float = 0.2
    w_lift: float = 0.1
    # w_goal must exceed (w_hold + w_lift) * number of goal conditions' share, or releasing an object
    # into place would lower Phi. With 2 conditions (task 0) each is worth w_goal/2 = 0.5 > 0.3.
    w_goal: float = 1.0
    invalid: float = -0.05
    repeat: float = -0.1
    per_decision: float = -0.005
    shaping: bool = True


def potential(cfg: PotentialConfig, progress: dict) -> float:
    return (cfg.w_hold * bool(progress.get('grasped_objects_of_interest'))
            + cfg.w_lift * bool(progress.get('lifted_objects_of_interest'))
            + cfg.w_goal * float(progress.get('fraction', 0.0)))


def potential_reward(cfg: PotentialConfig, *, prev_progress: dict, progress: dict, success: bool,
                     invalid: bool, repeated: bool) -> DecisionReward:
    shaping = potential(cfg, progress) - potential(cfg, prev_progress) if cfg.shaping else 0.0
    parts = {
        'success': cfg.success if success else 0.0,
        'shaping': shaping,
        'invalid': cfg.invalid if invalid else 0.0,
        'repeat': cfg.repeat if repeated else 0.0,
        'per_decision': cfg.per_decision,
    }
    return DecisionReward(sum(parts.values()), parts)


def decision_reward(cfg: RewardConfig, *, prev_fraction: float, fraction: float, success: bool,
                    first_grasp: bool, status: str, repeated: bool) -> DecisionReward:
    parts = {
        'success': cfg.success if success else 0.0,
        'progress': cfg.progress * (fraction - prev_fraction),   # signed: undoing progress costs reward
        'grasp': cfg.grasp if first_grasp else 0.0,
        'rejected': cfg.rejected if (status not in OK and status not in NEUTRAL) else 0.0,
        'repeat': cfg.repeat if repeated else 0.0,
        'per_decision': cfg.per_decision,
    }
    return DecisionReward(sum(parts.values()), parts)
