"""PPO for Muse Glimmer in the harness on LIBERO-90.

    python scripts/train_ppo.py --output runs/stage2 --max-decisions 3 \
        --demo-states data/libero90_starts.npz --demo-fraction 1.0

One update = collect ``--decisions-per-update`` decisions, compute GAE per
decision, then ``--epochs`` passes of clipped PPO over them. Only the LoRA
adapters and the value head train. Metrics go to <output>/metrics.jsonl,
adapters to <output>/checkpoints/update_XXXX.
"""
import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from glimmer_ppo.env import EnvConfig, HarnessEnv
from glimmer_ppo.policy import Policy, PolicyConfig
from glimmer_ppo.ppo import PPOConfig, explained_variance, gae, kl_penalty, policy_loss, value_loss
from glimmer_ppo.reward import RewardConfig
from glimmer_ppo.starts import StartSampler, parse_ids


def build_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, default=100)
    p.add_argument('--save-every', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    # where episodes start
    p.add_argument('--suite', default='libero_90')
    p.add_argument('--tasks', default='0-89')
    p.add_argument('--inits', default='0-49')
    p.add_argument('--demo-states', type=Path)
    p.add_argument('--demo-fraction', type=float, default=0.0)
    # episode limits
    p.add_argument('--max-decisions', type=int, default=80)
    p.add_argument('--max-steps', type=int, default=900)
    p.add_argument('--record-video', action='store_true')
    # model
    p.add_argument('--model', default=PolicyConfig.model)
    p.add_argument('--load-in-4bit', action='store_true')
    p.add_argument('--lora-rank', type=int, default=16)
    p.add_argument('--lora-alpha', type=int, default=32)
    p.add_argument('--max-new-tokens', type=int, default=2048)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--reasoning', default='high')
    # PPO
    for name, value in asdict(PPOConfig()).items():
        flag = '--' + name.replace('_', '-')
        if isinstance(value, bool):
            p.add_argument('--no-' + name.replace('_', '-'), dest=name, action='store_false')
        else:
            p.add_argument(flag, type=type(value) if value is not None else float, default=value)
    p.add_argument('--minibatch', type=int, default=16, help='decisions per optimizer step')
    return p.parse_args()


def main():
    args = build_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    ppo = PPOConfig(**{k: getattr(args, k) for k in asdict(PPOConfig())})
    pcfg = PolicyConfig(model=args.model, load_in_4bit=args.load_in_4bit, lora_rank=args.lora_rank,
                        lora_alpha=args.lora_alpha, max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, reasoning_strength=args.reasoning or None)
    ecfg = EnvConfig(max_decisions=args.max_decisions, max_steps=args.max_steps,
                     record_video=args.record_video, reward=RewardConfig())
    sampler = StartSampler(args.suite, parse_ids(args.tasks, 90 if args.suite == 'libero_90' else 10),
                           parse_ids(args.inits, 50), args.demo_states, args.demo_fraction, seed=args.seed)
    (out / 'config.json').write_text(json.dumps(dict(args={k: str(v) for k, v in vars(args).items()},
                                                     ppo=asdict(ppo), policy=asdict(pcfg), env=asdict(ecfg)), indent=2))

    policy = Policy(pcfg)
    lora = [p for p in policy.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([dict(params=lora, lr=ppo.lr),
                             dict(params=policy.value_head.parameters(), lr=ppo.value_lr or ppo.lr)])
    print(f'[ppo] LoRA on {policy.n_lora_targets} linear layers, '
          f'{sum(p.numel() for p in lora) / 1e6:.1f}M trainable + value head', flush=True)

    env = HarnessEnv(out / 'episodes', ecfg)
    messages = env.reset(sampler())
    episode_return, episode_log = 0.0, []

    for update in range(1, args.updates + 1):
        t0 = time.time()
        batch = []
        # ---- 1. collect -------------------------------------------------
        policy.eval()
        while len(batch) < ppo.decisions_per_update:
            text, truncated, rec = policy.act(messages)
            with torch.no_grad():
                logp_old, value = policy.score(rec)
                logp_ref, _ = policy.score(rec, with_value=False, reference=True)
            reward, done, info = env.step(text, truncated)
            rec.update(logp_old=logp_old.cpu(), value=float(value), reward=reward,
                       kl=float(kl_penalty(logp_old, logp_ref)), done=done, info=info)
            batch.append(rec)
            episode_return += reward
            if done:
                episode_log.append(dict(return_=episode_return, success=info['success'],
                                        decisions=info['decisions'], goal_fraction=info['goal_fraction'],
                                        start=env.start.label, task=env.start.task_id))
                episode_return = 0.0
                messages = env.reset(sampler())
            else:
                messages = env.messages
        last_value = 0.0 if batch[-1]['done'] else policy.value_of(messages)

        # ---- 2. advantages ---------------------------------------------
        rewards = [r['reward'] - ppo.kl_coef * r['kl'] for r in batch]
        values = [r['value'] for r in batch]
        adv, ret = gae(rewards, values, [r['done'] for r in batch], ppo.gamma, ppo.lam, last_value)
        if ppo.normalize_advantages and len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # ---- 3. PPO epochs ----------------------------------------------
        policy.train()
        warmup = update <= ppo.critic_warmup_updates
        stats = dict(clipfrac=[], approx_kl=[], policy_loss=[], value_loss=[])
        for _ in range(ppo.epochs):
            order = list(range(len(batch)))
            random.shuffle(order)
            for start in range(0, len(order), args.minibatch):
                idx = order[start:start + args.minibatch]
                opt.zero_grad(set_to_none=True)
                for i in idx:
                    rec = batch[i]
                    logp, value = policy.score(rec)
                    a = adv[i].to(logp.device)
                    pl, s = policy_loss(logp, rec['logp_old'].to(logp.device), a, ppo.clip)
                    vl = value_loss(value, torch.tensor(rec['value'], device=value.device),
                                    ret[i].to(value.device), ppo.value_clip)
                    loss = (0.0 if warmup else 1.0) * pl.to(vl.device) + ppo.vf_coef * vl
                    (loss / len(idx)).backward()
                    stats['clipfrac'].append(s['clipfrac']); stats['approx_kl'].append(s['approx_kl'])
                    stats['policy_loss'].append(float(pl)); stats['value_loss'].append(float(vl))
                torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), ppo.max_grad_norm)
                opt.step()

        # ---- 4. log -----------------------------------------------------
        mean = lambda xs: sum(xs) / len(xs) if xs else float('nan')
        done_eps, episode_log = episode_log, []
        row = dict(update=update, seconds=round(time.time() - t0, 1), critic_warmup=warmup,
                   decisions=len(batch), mean_reward=mean([r['reward'] for r in batch]),
                   mean_kl=mean([r['kl'] for r in batch]),
                   explained_variance=explained_variance(values, ret),
                   response_tokens=mean([len(r['response']) for r in batch]),
                   truncated=sum(r['info']['status'] == 'truncated' for r in batch),
                   rejected=sum(r['info']['reward_parts']['rejected'] != 0 for r in batch),
                   episodes=len(done_eps), success_rate=mean([e['success'] for e in done_eps]),
                   mean_goal_fraction=mean([e['goal_fraction'] for e in done_eps]),
                   **{k: mean(v) for k, v in stats.items()})
        with (out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        with (out / 'episodes.jsonl').open('a') as f:
            for e in done_eps:
                f.write(json.dumps(dict(update=update, **e)) + '\n')
        print(json.dumps(row), flush=True)
        if update % args.save_every == 0 or update == args.updates:
            policy.save(out / 'checkpoints' / f'update_{update:04d}')
    env.close()


if __name__ == '__main__':
    main()
