#!/usr/bin/env python3
"""Train Glimmer through CaPtain's Python harness with its physical critic.

Hardware adaptation: sequential G=4 rollouts, NF4 QLoRA, full-distribution HF
sampling, synchronous single-pass updates. Starts from base Glimmer, not Qwen SFT.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.request import urlopen

import torch
from tqdm import tqdm

from glimmer_ppo.captain_rl import (RequestReservoir, actor_loss, atomic_json, ensure_critic, grouped_credit,
                                    fresh_seeds, join_requests, jsonl)
from glimmer_ppo.policy import Policy, PolicyConfig
from captain_baseline import PROJECT, verify
from serve_glimmer_python import FrozenGlimmer, make_server


class TrainingBackend(FrozenGlimmer):
    frozen = False

    def __init__(self, args, policy):
        self.args, self.policy, self.torch = args, policy, torch
        self.model, self.processor = policy.model, policy.processor
        self.device = policy._device()
        self.lock, self.phase_lock, self.progress_lock = threading.Lock(), threading.Lock(), threading.Lock()
        self.reservoir = None
        self.pending = None
        self.status = dict(phase='initializing', round=0, rounds=args.rounds)
        self.token_at = 0
        self.cancel = threading.Event()

    def progress(self, **values):
        with self.progress_lock:
            self.status.update(values)
            atomic_json(self.args.output / 'progress.json', self.status)

    def token_progress(self, done, total):
        if self.cancel.is_set():
            raise InterruptedError('training rollout cancelled')
        now = time.monotonic()
        if done == 0 or now - self.token_at >= 1:
            self.token_at = now
            self.progress(phase='generating', generated_tokens=done, token_cap=total)

    def begin_episode(self, seed):
        with self.phase_lock:
            self.cancel.clear()
            self.reservoir = RequestReservoir(self.args.max_training_requests, seed)

    def end_episode(self):
        with self.phase_lock:
            result, self.reservoir = self.reservoir, None
            return result

    def capture_generation(self, inputs, tokens, result):
        rid = result['id']
        selected = self.reservoir.wants(rid, len(tokens))
        self.reservoir.metadata[-1].update(
            python_calls=len(result['choices'][0]['message'].get('tool_calls', [])),
            parse_errors=bool(result['glimmer_bridge']['parse_errors']))
        if selected:
            self.pending = (rid, dict(inputs={k: v.detach().cpu() if torch.is_tensor(v) else v
                                             for k, v in inputs.items()}, response=tokens.detach().cpu()))

    def generate(self, body):
        with self.phase_lock:
            if self.reservoir is None:
                raise ValueError('trainer is not collecting an episode; no generation during updates')
            if float(body.get('temperature', 1)) != 1:
                raise ValueError('this training run samples at temperature 1')
            self.pending = None
            response, raw = super().generate(body)
            if self.pending:
                rid, record = self.pending
                # Copy OUTSIDE inference_mode: autograd must be allowed to save these tensors.
                record = dict(inputs={k: v.clone() if torch.is_tensor(v) else v for k, v in record['inputs'].items()},
                              response=record['response'].clone())
                self.reservoir.add(rid, record)
                self.pending = None
            return response, raw


def load_upstream(checkout):
    verify(checkout)
    sys.path.insert(0, str(checkout / 'rl'))
    os.environ['RL_REWARD_VERSION'] = 'v12b'
    return tuple(importlib.import_module(name) for name in ('physics_critic', 'critic_cotrain', 'reward', 'code_reward'))


def append(path, row):
    with path.open('a') as stream:
        stream.write(json.dumps(row, default=str, allow_nan=False) + '\n')


def checkpoint(path, policy, actor_opt, critic, critic_opt, round_index, sigma, args):
    staging = path.with_name(path.name + '.tmp')
    staging.mkdir(parents=True)
    policy.save(str(staging))
    torch.save(dict(round=round_index, sigma=sigma, actor_optimizer=actor_opt.state_dict(),
                    critic=critic.state_dict(), critic_optimizer=critic_opt.state_dict(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    python_rng=random.getstate(), args=vars(args)), staging / 'trainer.pt')
    staging.rename(path)


def episode(args, backend, round_index, sample, env_seed, reward_module, code_module, features):
    harness = args.checkout / 'harness'
    root = args.output / 'rollouts' / f'round_{round_index:04d}' / f'sample_{sample:02d}'
    directory = root / f'{args.suite}_{args.task}_seed-{env_seed}'
    env = dict(os.environ, CAPTAIN_HARNESS=str(harness), CAPTAIN_PYTHON=args.harness_python,
               STUDENT_PYTHON=args.student_python, POLICY_URL=f'http://127.0.0.1:{args.port}/v1',
               POLICY_MODEL=args.served_name, TEMP='1.0', SAM3_URL=args.sam3_url, MOLMO_URL=args.molmo_url,
               EGL_DEVICE=str(args.egl_device))
    env['FRESH_ENV_PREFLIGHT_PATH'] = str(args.output / 'toolchain.json')
    # CUDA_VISIBLE_DEVICES on the learner may hide the perception/render GPU. The
    # simulator is a separate process; render with the requested physical EGL ID.
    env.pop('CUDA_VISIBLE_DEVICES', None)
    command = ['bash', str(harness / 'eval/run_episode.sh'), args.suite, str(args.task), str(env_seed), str(root),
               '--mark-requests', '--max-requests', str(args.max_requests), '--max-actions', str(args.max_actions),
               '--wall-time', str(args.wall_time), '--sandbox-hide', str(args.output),
               '--sandbox-hide', str(PROJECT / 'runs'), '--sandbox-hide', str(args.critic_dir),
               '--sandbox-hide', str(harness / 'data/libero_fresh_init')]
    backend.begin_episode(args.seed + round_index * args.group_size + sample)
    backend.progress(phase='collecting', round=round_index, sample=sample + 1, group_size=args.group_size,
                     start_seed=env_seed, generated_tokens=0, episode=str(directory))
    log = args.output / f'episode_round{round_index:04d}_sample{sample:02d}.log'
    try:
        with log.open('w') as stream:
            process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                process.wait(timeout=args.wall_time + 180)
            except BaseException:
                backend.cancel.set()
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except ProcessLookupError:
                    pass
                raise
    finally:
        capture = backend.end_episode()
    if not (directory / 'episode_result.json').is_file():
        raise RuntimeError(f'No episode result (exit {process.returncode}); see {log}')
    score = reward_module.score_episode(directory)
    if score.get('verdict') in ('infrastructure_error', 'replay_infidelity') or not score.get('diagnostics', {}).get('reward_valid'):
        raise RuntimeError(f'Infrastructure failure; no policy update: {score}; see {log}')
    driver = jsonl(directory / 'driver/requests.jsonl')
    indices = join_requests(driver, capture.metadata, capture.records)
    rows = features.episode_live_rows(directory)
    ids = features.driver_response_ids(directory)
    marked = {r['request_index'] for r in rows}
    physical = {indices[r['response_id']] for r in capture.metadata if r['python_calls']}
    if not physical.issubset(marked) or not marked.issubset(ids) or set(ids.values()) != set(indices):
        raise RuntimeError('Privileged marks do not cover every executed Python request; no update')
    labels = code_module.episode_request_costs(directory)
    if any(not label.get('scored') for label in labels.values()):
        raise RuntimeError('Unscored code execution/infrastructure label; no policy update')
    sample_path = directory / 'glimmer_capture.json'
    atomic_json(sample_path, dict(requests=capture.metadata, sampled_response_ids=list(capture.records)))
    score.update(suite=args.suite, task_id=args.task, episode_dir=str(directory), sample_path=str(sample_path))
    success = bool(score['diagnostics']['host_final_success'])
    row = dict(round=round_index, sample=sample, seed=env_seed, success=success,
               strict_success=success and score['diagnostics']['agent_finished'], requests=len(driver),
               episode_dir=str(directory), verdict=score['verdict'])
    append(args.output / 'episodes.jsonl', row)
    print('[captain-rl] episode ' + json.dumps(row), flush=True)
    return dict(score=score, capture=capture, labels=labels, indices=indices)


def optimize_group(policy, actor_opt, critic, critic_opt, cc, pc, trajectories, costs, sigma, args, progress):
    """All credit is frozen before updating either model; actor receives no critic gradients."""
    critic.eval()
    results = [item['score'] for item in trajectories]
    with torch.no_grad():
        gae, buf = cc.physics_only_advantages(critic, critic.featurizer, results,
            lam=.9, sigma=1., device='cpu', false_finish_reward=-.3, costs=costs)
    totals = []
    for result in results:
        rows = critic.featurizer.live_rows(result['episode_dir'])
        task_key = f'{result["suite"]}:{result["task_id"]}'
        totals.append(pc.staged_return(result, rows, task_key, -.3, costs) +
                      sum((pc.request_costs(rows, costs, task_key) or {}).values()))
    keyed = grouped_credit(gae, totals,
        [[r['response_id'] for r in t['capture'].metadata] for t in trajectories], max(sigma or 1., 1e-4))
    batches = []
    policy.eval()
    for index, trajectory in enumerate(trajectories):
        capture = trajectory['capture']
        if set(keyed[index]) != {r['response_id'] for r in capture.metadata}:
            raise RuntimeError('Response IDs and critic credit do not match')
        total_tokens = sum(r['tokens'] for r in capture.metadata)
        inclusion = len(capture.metadata) / len(capture.records)
        for rid, record in capture.records.items():
            progress(phase='scoring behavior', scored=len(batches), trained_requests=sum(len(t['capture'].records) for t in trajectories))
            with torch.no_grad():
                old, _ = policy.score(record, with_value=False)
                reference, _ = policy.score(record, with_value=False, reference=True)
            label = trajectory['labels'].get(trajectory['indices'][rid], {})
            metadata = next(r for r in capture.metadata if r['response_id'] == rid)
            batches.append(dict(record=record, old=old.detach().cpu(), reference=reference.detach().cpu(),
                                advantage=keyed[index][rid], pure_error=bool(label.get('no_effect') or metadata.get('parse_errors')),
                                code_cost=-1. if metadata.get('parse_errors') else float(label.get('cost') or 0), inclusion=inclusion,
                                trajectory_tokens=total_tokens))
    actor_opt.zero_grad(set_to_none=True)
    policy.train()
    stats = []
    for i, batch in enumerate(tqdm(batches, desc='Glimmer RL backward', unit='request')):
        logp, _ = policy.score(batch['record'], with_value=False)
        loss, row = actor_loss(logp, batch['old'], batch['reference'], batch['advantage'],
            pure_error=batch['pure_error'], code_cost=batch['code_cost'], inclusion=batch['inclusion'],
            trajectory_tokens=batch['trajectory_tokens'], trajectories=len(trajectories),
            horizon=args.max_requests, kl_beta=args.kl_beta)
        loss.backward()
        stats.append(row)
        progress(phase='backward', backward=i + 1, backward_total=len(batches), **row)
    norm = torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), 1.)
    if not torch.isfinite(norm):
        actor_opt.zero_grad(set_to_none=True)
        raise FloatingPointError('Nonfinite actor gradients; optimizer not stepped')
    stepped = float(norm) > 0
    if stepped:
        actor_opt.step()
    actor_opt.zero_grad(set_to_none=True)
    policy.eval()
    # The physical critic is separate and remains in eval mode (no stochastic
    # dropout) while its BCE gradients are fitted, as in the upstream trainer.
    xs = torch.cat(buf['x']) if buf['x'] else torch.empty((0, len(critic.featurizer.features) + len(critic.featurizer.tasks)))
    ys = torch.tensor(buf['y'], dtype=torch.float32)
    critic_losses = cc.update(critic, critic_opt, xs, None, ys, steps=2) if len(ys) else []
    advantages = [v * max(sigma or 1., 1e-4) for row in keyed for v in row.values()]
    average = sum(advantages) / len(advantages)
    observed = math.sqrt(sum((x - average) ** 2 for x in advantages) / len(advantages))
    sigma = cc.ema_std(sigma, observed or 1.)
    means = {k: sum(row[k] for row in stats) / len(stats) for k in stats[0]}
    return sigma, dict(**means, critic_bce=critic_losses[-1] if critic_losses else None, advantage_std=observed,
                       actor_grad_norm=float(norm), actor_optimizer_stepped=stepped, trained_requests=len(batches),
                       no_physical_mark_episodes=sum(not a for a in gae), group_returns=totals,
                       cost_stats=buf.get('cost_stats'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkout', type=Path, default=PROJECT / 'data/captain/upstream')
    p.add_argument('--critic-dir', type=Path, default=PROJECT / 'data/captain/critic/warm_start')
    p.add_argument('--model', default=PolicyConfig.model)
    p.add_argument('--suite', default='libero_10', choices=['libero_10'])
    p.add_argument('--task', type=int, default=0, choices=[0])
    p.add_argument('--rounds', type=int, default=5, help='total target rounds, including resumed rounds')
    p.add_argument('--group-size', type=int, default=4, choices=[4])
    p.add_argument('--max-training-requests', type=int, default=8)
    p.add_argument('--max-requests', type=int, default=60)
    p.add_argument('--max-actions', type=int, default=120)
    p.add_argument('--wall-time', type=int, default=1080)
    p.add_argument('--context-tokens', type=int, default=65536)
    p.add_argument('--max-new-tokens', type=int, default=4096)
    p.add_argument('--load-in-4bit', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--reasoning', choices=['low','medium','high'], default='high')
    p.add_argument('--lr', type=float, default=4.24e-6)
    p.add_argument('--kl-beta', type=float, default=.002)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--port', type=int, default=8601)
    p.add_argument('--served-name', default='glimmer-rl')
    p.add_argument('--harness-python')
    p.add_argument('--student-python', default=os.environ.get('STUDENT_PYTHON'))
    p.add_argument('--sam3-url', default='http://127.0.0.1:8114')
    p.add_argument('--molmo-url', default='http://127.0.0.1:8122/v1')
    p.add_argument('--egl-device', type=int, default=3)
    p.add_argument('--resume', type=Path)
    args = p.parse_args()
    args.output, args.checkout = args.output.resolve(), args.checkout.resolve()
    args.critic_dir = args.critic_dir.resolve()
    args.harness_python = args.harness_python or str(args.checkout / 'harness/.venv-libero/bin/python')
    args.student_python = args.student_python or str(args.checkout / 'harness/.venv-student/bin/python')
    if args.output.exists() and any(p.name != 'train.log' for p in args.output.iterdir()):
        p.error('use a fresh output folder, including when resuming')
    if args.rounds < 1 or args.max_training_requests < 1 or args.context_tokens <= args.max_new_tokens:
        p.error('invalid rounds, request sample size or context budget')
    # Mandatory startup validation is part of training; no separate check run.
    pc, cc, reward, code = load_upstream(args.checkout)
    if not Path(args.harness_python).is_file() or not Path(args.student_python).is_file():
        p.error('install the CaPtain simulator and student Python environments first; see docs/captain_rl.md')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / 'progress.json', dict(phase='initializing', round=0, rounds=args.rounds))
    (args.output / 'train.pid').write_text(str(os.getpid()) + '\n')
    config = dict(mode='captain_rl', args=vars(args), method='upstream physical critic + same-time GAE + grouped outcome + SAO ratio mask',
                  adaptations=['Glimmer base (no in-harness SFT initialization)', 'NF4 QLoRA', 'sequential G=4',
                               'one start group per optimizer step', 'full-distribution sampling, no top-k correction',
                               'base Glimmer KL reference', 'no critic-based early termination'],
                  critic_initialization='Released Qwen-lineage critic; transfer calibration unverified, co-trained on Glimmer outcomes')
    atomic_json(args.output / 'config.json', config)
    server, thread, backend = None, None, None
    try:
        for url, marker in ((args.sam3_url.rstrip('/') + '/openapi.json', 'sam'),
                            (args.molmo_url.rstrip('/') + '/models', 'molmo')):
            with urlopen(url, timeout=10) as response:
                service = json.load(response)
            if marker == 'sam' and not {'/segment','/segment_point'}.issubset(service.get('paths', {})):
                raise RuntimeError('SAM endpoint does not provide the expected segmentation API')
            if marker == 'molmo' and not any('molmo' in str(row.get('id', '')).lower() for row in service.get('data', [])):
                raise RuntimeError('pointing endpoint is not a served Molmo model')
        # Required upstream runtime validation is integrated into training startup,
        # not a separate experiment or robot-evaluation run.
        startup = subprocess.run([args.harness_python, str(args.checkout / 'rl/toolchain_preflight.py'),
            '--harness', str(args.checkout / 'harness'), '--contract', 'python-runtime/v1',
            '--student-python', args.student_python], capture_output=True, text=True, timeout=240)
        if startup.returncode:
            raise RuntimeError('Training startup environment invalid: ' + (startup.stdout + startup.stderr)[-4000:])
        atomic_json(args.output / 'toolchain.json', json.loads(startup.stdout))
        ensure_critic(args.critic_dir)
        starts, bank = fresh_seeds(args.checkout / 'harness', args.suite, args.task, args.seed)
        config['state_bank'] = bank
        atomic_json(args.output / 'config.json', config)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        critic = cc.CoCritic(args.critic_dir, hidden_dim=0, use_hidden=False).cpu().eval()
        if critic.featurizer.same_t != 'req_idx' or len(critic.featurizer.features) != 37:
            raise RuntimeError('expected released physical-only same-time critic')
        critic_opt = torch.optim.AdamW(critic.parameters(), lr=1e-4)
        policy_config = PolicyConfig(model=args.model, load_in_4bit=args.load_in_4bit,
                                     reasoning_strength=args.reasoning, lora_dropout=0.,
                                     activation_offload=True, max_new_tokens=args.max_new_tokens,
                                     max_prompt_tokens=None)
        config['policy'] = asdict(policy_config)
        atomic_json(args.output / 'config.json', config)
        policy = Policy(policy_config)
        policy.value_head.requires_grad_(False)
        policy.eval()
        actor_opt = torch.optim.AdamW(policy.trainable_parameters(), lr=args.lr)
        first, sigma = 1, None
        if args.resume:
            saved = torch.load(args.resume / 'trainer.pt', map_location='cpu', weights_only=False)
            for key in ('model','load_in_4bit','suite','task','seed','group_size','max_requests','max_actions','lr','kl_beta','reasoning','context_tokens','max_new_tokens','max_training_requests','wall_time'):
                if saved['args'][key] != vars(args)[key]:
                    raise ValueError(f'resume configuration changed: {key}')
            policy.load(args.resume)
            actor_opt.load_state_dict(saved['actor_optimizer'])
            critic.load_state_dict(saved['critic'])
            critic_opt.load_state_dict(saved['critic_optimizer'])
            torch.set_rng_state(saved['torch_rng'])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(saved['cuda_rng'])
            random.setstate(saved['python_rng'])
            first, sigma = saved['round'] + 1, saved['sigma']
        if first > args.rounds:
            raise ValueError('--rounds must exceed the resumed checkpoint round')
        costs = dict(c_act=.008, c_fail=.02, missed_finish_return=.25, exhaust_reward=-.1,
                     post_success_act=True, post_success_req=.005, post_success_cap=.45,
                     ref=json.loads((args.checkout / 'rl/configs/act-ref.json').read_text())['ref'])
        backend = TrainingBackend(args, policy)
        server = make_server(backend, '127.0.0.1', args.port, args.output)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        for round_index in tqdm(range(first, args.rounds + 1), desc='Glimmer RL rounds', unit='round'):
            t0 = time.monotonic()
            trajectories = []
            for sample in range(args.group_size):
                trajectories.append(episode(args, backend, round_index, sample,
                    starts[(round_index - 1) % len(starts)], reward, code, pc))
            sigma, metrics = optimize_group(policy, actor_opt, critic, critic_opt, cc, pc, trajectories,
                                           costs, sigma, args, backend.progress)
            successes = sum(t['score']['diagnostics']['host_final_success'] for t in trajectories)
            row = dict(round=round_index, seconds=round(time.monotonic() - t0, 1),
                       episodes=args.group_size, successes=successes, success_rate=successes / args.group_size,
                       mean_return=sum(metrics['group_returns']) / len(metrics['group_returns']),
                       requests=sum(len(t['capture'].metadata) for t in trajectories), **metrics)
            append(args.output / 'metrics.jsonl', row)
            backend.progress(phase='saving checkpoint', round=round_index)
            path = args.output / 'checkpoints' / f'round_{round_index:04d}'
            path.parent.mkdir(exist_ok=True)
            checkpoint(path, policy, actor_opt, critic, critic_opt, round_index, sigma, args)
            print('[captain-rl] checkpoint saved: ' + str(path), flush=True)
            print('[captain-rl] ' + json.dumps(row), flush=True)
            del trajectories
        backend.progress(phase='complete', round=args.rounds)
    except BaseException as exc:
        atomic_json(args.output / 'progress.json', dict(phase='interrupted' if isinstance(exc, KeyboardInterrupt) else 'error',
                    error=f'{type(exc).__name__}: {exc}', rounds=args.rounds))
        raise
    finally:
        if server:
            server.shutdown()
            server.server_close()
            thread.join(timeout=10)


if __name__ == '__main__':
    main()
