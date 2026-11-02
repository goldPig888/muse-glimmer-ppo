"""PPO for Muse Glimmer.

Paper protocol, LIBERO-10 task 0 (the first task of the paper), training on
initial states 10-49 and keeping 0-9 for evaluation (scripts/eval_task.py):

    python scripts/train_ppo.py --output runs/task0 --env paper --task 0 \
        --train-inits 10-49 --max-decisions 80

Harness actions on LIBERO-90 (later):

    python scripts/train_ppo.py --output runs/stage2 --env harness --max-decisions 3 \
        --demo-states data/libero90_starts.npz --demo-fraction 1.0

One update = collect ``--decisions-per-update`` decisions, compute GAE per
decision, then ``--epochs`` passes of clipped PPO over them. Only the LoRA
adapters and the value head train. Metrics go to <output>/metrics.jsonl,
adapters to <output>/checkpoints/update_XXXX.
"""
import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from tqdm.auto import tqdm

from glimmer_ppo.paper_env import PaperEnv, PaperEnvConfig
from glimmer_ppo.checkpoint import checkpoint_config, checkpoint_update, restore_checkpoint, save_checkpoint
from glimmer_ppo.policy import Policy, PolicyConfig
from glimmer_ppo.ppo import PPOConfig, explained_variance, gae, kl_penalty, policy_loss, value_loss
from glimmer_ppo.reward import PotentialConfig, RewardConfig
from glimmer_ppo.rollout_server import LlamaServer, ServerConfig, generate_with_server
from glimmer_ppo.starts import StartSampler, parse_ids
from glimmer_ppo.assistance import AssistedStartSampler, AssistanceConfig, AssistanceCurriculum


def build_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', type=Path, help='checkpoint directory; continue in a fresh --output folder')
    p.add_argument('--init-policy', type=Path,
                   help='initialize adapter/value weights only (e.g. SFT); start fresh PPO optimizer and updates')
    p.add_argument('--updates', type=int, default=100, help='total target update number, including resumed updates')
    p.add_argument('--save-every', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--no-progress', action='store_true', help='disable terminal progress bars (status lines remain)')
    p.add_argument('--no-token-progress', action='store_true', help='disable the per-token HF generation bar')
    p.add_argument('--env', choices=['paper', 'harness'], default='paper',
                   help='paper: move_to/done/give_up as in the paper; harness: skill programs')
    # where episodes start
    p.add_argument('--task', type=int, default=0, help='paper env: LIBERO-10 task id')
    p.add_argument('--train-inits', default='10-49', help='paper env: initial states to train on (0-9 held out)')
    p.add_argument('--reward', choices=['sparse', 'subgoal', 'both'], default='both',
                   help='paper env reward ablation: success+costs only / + sub-goal potential / + dense distances')
    p.add_argument('--suite', default='libero_90', help='harness env')
    p.add_argument('--tasks', default='0-89', help='harness env')
    p.add_argument('--inits', default='0-49', help='harness env')
    p.add_argument('--demo-states', type=Path)
    p.add_argument('--demo-fraction', type=float, default=0.0)
    p.add_argument('--demo-kinds', default='pre_grasp,pre_release',
                   help='comma-separated demo start types; use pre_release for the last-placement stage')
    p.add_argument('--object-references', type=Path,
                   help='full-demo reference directory; enable privileged object-force curriculum')
    p.add_argument('--assist-gain', type=float, default=1.)
    p.add_argument('--assist-decay', type=float, default=.8)
    p.add_argument('--assist-window', type=int, default=5, help='completed episodes required at each gain level')
    p.add_argument('--assist-minimum', type=float, default=.05)
    p.add_argument('--assist-force', type=float, default=3., help='maximum assistance per object in N at gain=1')
    p.add_argument('--assist-torque', type=float, default=.10, help='maximum torque in Nm at gain=1')
    p.add_argument('--assist-object-threshold', type=float, default=.8)
    p.add_argument('--assist-motion-threshold', type=float, default=.6)
    p.add_argument('--assist-contact-threshold', type=float, default=.5)
    p.add_argument('--assist-success-threshold', type=float, default=.8)
    # episode limits
    p.add_argument('--max-decisions', type=int, default=80)
    p.add_argument('--max-steps', type=int, default=900)
    p.add_argument('--record-video', action='store_true')
    # model
    p.add_argument('--model', default=PolicyConfig.model)
    p.add_argument('--load-in-4bit', action='store_true')
    p.add_argument('--activation-offload', action='store_true',
                   help='save backward activations in CPU RAM to fit long contexts on smaller GPUs')
    p.add_argument('--max-prompt-tokens', type=int,
                   help='bound context by dropping the oldest complete turns; keep task and recent observations')
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
    # rollout engine
    p.add_argument('--rollout', choices=['hf', 'llamacpp'], default='hf',
                   help='hf: generate with the training model; llamacpp: generate on a llama-server (other GPU)')
    p.add_argument('--llama-server', help='path to llama-server')
    p.add_argument('--gguf', help='Muse GGUF for rollouts (Q8_0 recommended)')
    p.add_argument('--mmproj', help='Muse vision projector GGUF')
    p.add_argument('--llama-convert', help="path to llama.cpp's convert_lora_to_gguf.py")
    p.add_argument('--hf-base-dir', help='local HF snapshot of the base model (default: from the HF cache)')
    p.add_argument('--server-gpu', default='0', help='physical GPU id for llama-server')
    p.add_argument('--server-port', type=int, default=8090)
    p.add_argument('--server-ctx', type=int, default=32768)
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--resume', type=Path)
    resume, _ = probe.parse_known_args()
    if resume.resume:
        try:
            saved = checkpoint_config(resume.resume)['args']
            defaults = {}
            for action in p._actions:
                if action.dest in {'output', 'resume', 'init_policy', 'help', 'no_progress', 'no_token_progress'}:
                    continue
                if action.dest not in saved:
                    continue
                value = saved[action.dest]
                if value is None or value == 'None':
                    value = None
                elif isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                    value = value is True or value == 'True'
                elif action.type is not None:
                    value = action.type(value)
                defaults[action.dest] = value
            p.set_defaults(**defaults)
        except (OSError, ValueError, KeyError) as exc:
            p.error(str(exc))
    args = p.parse_args()
    if args.object_references and (args.env != 'paper' or args.demo_states or args.demo_fraction):
        p.error('--object-references requires paper mode and cannot be mixed with mid-task demo starts')
    if args.resume and args.init_policy:
        p.error('--resume and --init-policy are mutually exclusive')
    if args.init_policy:
        try:
            saved_policy = checkpoint_config(args.init_policy)['policy']
            for name, value in (('model', args.model), ('lora_rank', args.lora_rank), ('lora_alpha', args.lora_alpha)):
                if saved_policy[name] != value:
                    p.error(f'--init-policy requires --{name.replace("_", "-")} {saved_policy[name]}')
        except (OSError, ValueError, KeyError) as exc:
            p.error(str(exc))
    if (args.output / 'config.json').exists():
        p.error('--output already contains a run; choose a fresh folder to preserve its data')
    if args.resume and checkpoint_update(args.resume) >= args.updates:
        p.error('--updates must exceed the checkpoint update number')
    return args


def main():
    args = build_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    ppo = PPOConfig(**{k: getattr(args, k) for k in asdict(PPOConfig())})
    pcfg = PolicyConfig(model=args.model, load_in_4bit=args.load_in_4bit,
                        activation_offload=args.activation_offload, lora_rank=args.lora_rank,
                        max_prompt_tokens=args.max_prompt_tokens,
                        token_progress=not args.no_progress and not args.no_token_progress,
                        lora_alpha=args.lora_alpha, max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature, reasoning_strength=args.reasoning or None)
    if args.env == 'paper':
        train_inits = parse_ids(args.train_inits, 50)
        if set(train_inits) & set(range(10)):
            print('[ppo] warning: training initial states overlap the 0-9 evaluation states', flush=True)
        ecfg = PaperEnvConfig(max_calls=args.max_decisions, max_steps=args.max_steps,
                              reward=PotentialConfig(mode=args.reward))
        sampler = StartSampler('libero_10', [args.task], train_inits, args.demo_states, args.demo_fraction,
                               kinds=tuple(args.demo_kinds.split(',')), seed=args.seed, allow_test_suite=True)
        if args.object_references:
            sampler = AssistedStartSampler(args.object_references, args.task,
                AssistanceConfig(gain=args.assist_gain, max_force=args.assist_force, max_torque=args.assist_torque),
                AssistanceCurriculum(gain=args.assist_gain, decay=args.assist_decay, window=args.assist_window,
                    minimum=args.assist_minimum, task_threshold=args.assist_object_threshold,
                    motion_threshold=args.assist_motion_threshold, contact_threshold=args.assist_contact_threshold,
                    success_threshold=args.assist_success_threshold),
                seed=args.seed)
    else:
        # Paper-mode installs do not need the optional legacy harness package.
        from glimmer_ppo.env import EnvConfig, HarnessEnv
        ecfg = EnvConfig(max_decisions=args.max_decisions, max_steps=args.max_steps,
                         record_video=args.record_video, reward=RewardConfig())
        sampler = StartSampler(args.suite, parse_ids(args.tasks, 90 if args.suite == 'libero_90' else 10),
                               parse_ids(args.inits, 50), args.demo_states, args.demo_fraction,
                               kinds=tuple(args.demo_kinds.split(',')), seed=args.seed)
    config = dict(resume_from_update=checkpoint_update(args.resume) if args.resume else 0,
                  args={k: str(v) for k, v in vars(args).items()},
                  ppo=asdict(ppo), policy=asdict(pcfg), env=asdict(ecfg))
    if args.object_references:
        config['object_assistance'] = dict(config=asdict(sampler.config),
            manifest_hash=sampler.manifest_hash, initial_curriculum=sampler.curriculum.state_dict(),
            start_kind='full_demo_initial_state', privileged_training=True)
    (out / 'config.json').write_text(json.dumps(config, indent=2))

    (out / 'train.pid').write_text(str(os.getpid()) + '\n')
    log = lambda message: tqdm.write(message)
    log(f'[ppo] output={out.resolve()} pid={os.getpid()}')
    if args.object_references:
        log(f'[voc] full-task demo starts; gain={sampler.curriculum.gain:.4f}; '
            'assisted successes are not unassisted evaluation')
    log(f'[ppo] loading {args.model}; 4bit={args.load_in_4bit} '
        f'activation_offload={args.activation_offload} max_prompt_tokens={args.max_prompt_tokens}')
    policy = Policy(pcfg)
    lora = [p for p in policy.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([dict(params=lora, lr=ppo.lr),
                             dict(params=policy.value_head.parameters(), lr=ppo.value_lr or ppo.lr)])
    completed_updates = 0
    if args.resume:
        completed_updates = restore_checkpoint(policy, opt, sampler, args.resume, log)
        log(f'[ppo] continuing from update {completed_updates}; next update={completed_updates + 1}. '
            'A fresh simulator episode starts; unfinished rollouts are not restored.')
    elif args.init_policy:
        policy.load(args.init_policy)
        log('[ppo] initialized policy weights; fresh optimizer, update counter and critic warmup')
    log(f'[ppo] LoRA on {policy.n_lora_targets} linear layers, '
        f'{sum(p.numel() for p in lora) / 1e6:.1f}M trainable + value head')

    env = PaperEnv(out / 'episodes', ecfg) if args.env == 'paper' else HarnessEnv(out / 'episodes', ecfg)
    tools = lambda: getattr(env, 'tools', None)

    server = None
    if args.rollout == 'llamacpp':
        missing = [f for f in ('llama_server', 'gguf', 'mmproj', 'llama_convert') if not getattr(args, f)]
        if missing:
            raise SystemExit('--rollout llamacpp needs ' + ', '.join('--' + m.replace('_', '-') for m in missing))
        base_dir = args.hf_base_dir
        if base_dir is None:
            from huggingface_hub import snapshot_download
            base_dir = snapshot_download(args.model, local_files_only=True)
        server = LlamaServer(ServerConfig(args.llama_server, args.gguf, args.mmproj, gpu=args.server_gpu,
                                          port=args.server_port, ctx=args.server_ctx,
                                          convert_script=args.llama_convert, hf_base_dir=base_dir),
                             out / 'server').start()
        if args.resume or args.init_policy:
            adapter_dir = out / 'server' / 'adapter_latest'
            policy.save_adapter(adapter_dir)
            server.load_adapter(adapter_dir)
        print(f'[ppo] rollouts on llama-server (GPU {args.server_gpu}), training on this process\'s GPUs', flush=True)

    def act(messages):
        if server is not None:
            return generate_with_server(server, policy, messages, tools())
        return policy.act(messages, tools())
    log('[ppo] starting simulator and resetting first episode')
    messages = env.reset(sampler())
    episode_return, episode_log = 0.0, []

    for update in tqdm(range(completed_updates + 1, args.updates + 1), total=args.updates,
                       initial=completed_updates, desc='PPO updates', unit='update',
                       position=0, dynamic_ncols=True, disable=args.no_progress):
        t0 = time.time()
        batch = []
        # ---- 1. collect -------------------------------------------------
        policy.eval()
        log(f'[ppo] update {update}/{args.updates}: collecting {ppo.decisions_per_update} decisions')
        rollout_bar = tqdm(total=ppo.decisions_per_update, desc='Rollout decisions', unit='decision',
                           position=1, leave=False, dynamic_ncols=True, disable=args.no_progress)
        while len(batch) < ppo.decisions_per_update:
            log(f'[ppo] update={update} decision={len(batch) + 1}/{ppo.decisions_per_update}: generating reply')
            text, truncated, rec = act(messages)
            log(f'[ppo] update={update} decision={len(batch) + 1}/{ppo.decisions_per_update} '
                  f'prompt_tokens={rec["inputs"]["input_ids"].shape[1]} '
                  f'response_tokens={len(rec["response"])}: scoring policy and reference')
            with torch.no_grad():
                logp_old, value = policy.score(rec)
                logp_ref, _ = policy.score(rec, with_value=False, reference=True)
            reward, done, info = env.step(text, truncated)
            rec.update(logp_old=logp_old.cpu(), value=float(value), reward=reward,
                       kl=float(kl_penalty(logp_old, logp_ref)), done=done, info=info)
            batch.append(rec)
            rollout_bar.update(1)
            rollout_bar.set_postfix(reward=f'{reward:.4f}', status=info['status'])
            log(f'[ppo] action={info.get("tool", "reply")} status={info["status"]} '
                f'reward={reward:+.4f} sim_steps={info.get("steps", "?")} done={done}')
            episode_return += reward
            if done:
                episode_log.append(dict(return_=episode_return, success=info['success'],
                                        decisions=info['decisions'], goal_fraction=info['goal_fraction'],
                                        start=env.start.label, task=env.start.task_id))
                if args.object_references:
                    episode_log[-1]['voc'] = info['voc']
                    sampler.curriculum.observe(dict(info['voc'], success=info['success']))
                episode_return = 0.0
                messages = env.reset(sampler())
            else:
                messages = env.messages
        rollout_bar.close()
        log(f'[ppo] update {update}: computing advantages')
        last_value = 0.0 if batch[-1]['done'] else policy.value_of(messages, tools())

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
        log(f'[ppo] update {update}: optimizing ({"critic warmup" if warmup else "actor + critic"})')
        optim_bar = tqdm(total=ppo.epochs * len(batch), desc='PPO backward passes', unit='decision',
                         position=1, leave=False, dynamic_ncols=True, disable=args.no_progress)
        for epoch in range(ppo.epochs):
            order = list(range(len(batch)))
            random.shuffle(order)
            for start in range(0, len(order), args.minibatch):
                idx = order[start:start + args.minibatch]
                log(f'[ppo] epoch={epoch + 1}/{ppo.epochs} '
                    f'minibatch={start // args.minibatch + 1}: backward {len(idx)} decisions')
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
                    stats['policy_loss'].append(float(pl.detach())); stats['value_loss'].append(float(vl.detach()))
                    optim_bar.update(1)
                    optim_bar.set_postfix(value_loss=f'{float(vl.detach()):.4f}')
                torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), ppo.max_grad_norm)
                opt.step()
        optim_bar.close()

        # ---- 4. log -----------------------------------------------------
        mean = lambda xs: sum(xs) / len(xs) if xs else float('nan')
        done_eps, episode_log = episode_log, []
        row = dict(update=update, seconds=round(time.time() - t0, 1), critic_warmup=warmup,
                   decisions=len(batch), mean_reward=mean([r['reward'] for r in batch]),
                   mean_kl=mean([r['kl'] for r in batch]),
                   explained_variance=explained_variance(values, ret),
                   response_tokens=mean([len(r['response']) for r in batch]),
                   truncated=sum(r['info']['status'] == 'truncated' for r in batch),
                   invalid=sum(r['info']['status'] in ('invalid', 'plan_error', 'COLLISION', 'IK_FAILURE')
                               or r['info']['reward_parts'].get('rejected', 0) != 0 for r in batch),
                   episodes=len(done_eps), success_rate=mean([e['success'] for e in done_eps]),
                   mean_goal_fraction=mean([e['goal_fraction'] for e in done_eps]),
                   **{k: mean(v) for k, v in stats.items()})
        if args.object_references:
            changed = sampler.curriculum.advance()
            row['assistance_next_gain'] = sampler.curriculum.gain
            row['assistance_batch_gain'] = mean([r['info']['voc']['gain'] for r in batch])
            row['assisted_episodes'] = sum(e['voc']['assisted'] for e in done_eps)
            for key in ('object_score', 'motion_score', 'contact_score', 'force_rms_N', 'impulse_Ns'):
                row['voc_' + key] = mean([e['voc'][key] for e in done_eps])
            if changed:
                log(f'[voc] tracking thresholds met: next episode gain={sampler.curriculum.gain:.4f}; '
                    'current episode retains its reset gain')
        with (out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        with (out / 'episodes.jsonl').open('a') as f:
            for e in done_eps:
                f.write(json.dumps(dict(update=update, **e)) + '\n')
        log(json.dumps(row))
        if update % args.save_every == 0 or update == args.updates:
            checkpoint = out / 'checkpoints' / f'update_{update:04d}'
            log(f'[ppo] saving checkpoint: {checkpoint}')
            save_checkpoint(policy, opt, sampler, checkpoint, update, config)
            log('[ppo] checkpoint saved')
        if server is not None:
            # Rollouts must come from the updated policy: push the new LoRA to llama.cpp.
            t1 = time.time()
            adapter_dir = out / 'server' / 'adapter_latest'
            policy.save_adapter(adapter_dir)
            server.load_adapter(adapter_dir)
            print(f'[ppo] llama-server reloaded with update {update} adapter in {time.time() - t1:.0f}s', flush=True)
    env.close()
    if server is not None:
        server.stop()
    log('[ppo] training complete')


if __name__ == '__main__':
    main()
