#!/usr/bin/env python3
"""Prepare and evaluate the released CaPtain baseline without changing its harness.

Uses only the standard library. Install the upstream environments separately;
see docs/captain_baseline.md. No Muse weights, PPO rewards or assistance are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.request import urlopen


PIPELINE_REV = '4bb86546196b80aaa2fa4bd0f9fa697aa7580e8e'
HARNESS_REV = 'de38d6d63b710caf356320a3c046b72b6d299514'
URL = 'https://github.com/waltstephen/CaPtain.git'
PROJECT = Path(__file__).resolve().parents[1]


def call(*args, **kwargs):
    return subprocess.run([str(a) for a in args], check=True, **kwargs)


def git(repo, *args):
    return call('git', '-C', repo, *args, capture_output=True, text=True).stdout.strip()


def verify(repo):
    for path, expected in ((repo, PIPELINE_REV), (repo / 'harness', HARNESS_REV)):
        if git(path, 'rev-parse', 'HEAD') != expected:
            raise ValueError(f'{path}: expected revision {expected}; use a separate pinned checkout')
        if git(path, 'status', '--porcelain', '--untracked-files=no'):
            raise ValueError(f'{path}: tracked files changed; baseline requires the unchanged release')


def prepare(args):
    if not args.checkout.exists():
        args.checkout.parent.mkdir(parents=True, exist_ok=True)
        call('git', 'clone', '--no-checkout', URL, args.checkout)
        call('git', '-C', args.checkout, 'checkout', '--detach', PIPELINE_REV)
        call('git', '-C', args.checkout, 'submodule', 'update', '--init', '--recursive')
    verify(args.checkout)
    source = args.checkout / 'harness/eval/tasks/libero_official_240.tsv'
    rows = [line for line in source.read_text().splitlines()
            if line and not line.startswith('#')]
    if not args.full:
        rows = [line for line in rows
                if line.split('\t')[4:6] == [args.suite, str(args.task)]]
    if not rows:
        raise ValueError('no matching official evaluation tasks')
    tasks = '# Unmodified rows from pinned CaPtain official held-out list\n' + '\n'.join(rows) + '\n'
    manifest = dict(pipeline_url=URL, pipeline_revision=PIPELINE_REV,
                    harness_revision=HARNESS_REV, episodes=len(rows),
                    tasks_sha256=hashlib.sha256(tasks.encode()).hexdigest(),
                    official_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    evaluation_only=True, assistance=False, custom_rewards=False,
                    scope='official_240' if args.full else f'{args.suite}/task{args.task}')
    args.plan.mkdir(parents=True, exist_ok=True)
    for path, value in ((args.plan / 'tasks.tsv', tasks),
                        (args.plan / 'manifest.json', json.dumps(manifest, indent=2) + '\n')):
        if path.exists() and path.read_text() != value:
            raise ValueError(f'{path} differs; choose a fresh --plan directory')
        path.write_text(value)
    print(json.dumps(manifest, indent=2))
    print(f'Prepared {len(rows)} episodes: {args.plan / "tasks.tsv"}', flush=True)


def plan_check(args):
    verify(args.checkout)
    manifest = json.loads((args.plan / 'manifest.json').read_text())
    digest = hashlib.sha256((args.plan / 'tasks.tsv').read_bytes()).hexdigest()
    if digest != manifest['tasks_sha256']:
        raise ValueError('task list changed since prepare')
    if (manifest['pipeline_revision'], manifest['harness_revision']) != (PIPELINE_REV, HARNESS_REV):
        raise ValueError('plan revisions do not match this launcher')
    return manifest


def readiness(args):
    """Basic readiness, not a substitute for an end-to-end perception/physics test."""
    problems, report = [], {}
    harness_python = os.environ.get('CAPTAIN_PYTHON', str(args.checkout / 'harness/.venv-libero/bin/python'))
    for label, executable, code in (
        ('harness_python', harness_python,
         'import sys, captain_harness, libero, mujoco; '
         'assert sys.version_info[:2] == (3,10), "requires Python 3.10"; print(sys.version)'),
        ('student_python', os.environ.get('STUDENT_PYTHON', ''),
         'import sys,numpy,scipy,PIL; assert sys.version_info[:3] == (3,11,16), "requires Python 3.11.16"; '
         'assert (numpy.__version__,scipy.__version__,PIL.__version__) == ("1.26.4","1.13.1","10.4.0"); print(sys.version)'),
    ):
        try:
            if not executable:
                raise ValueError('set STUDENT_PYTHON to the sandbox Python 3.11.16 interpreter')
            report[label] = call(executable, '-c', code, capture_output=True, text=True,
                                 timeout=45).stdout.strip()
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            problems.append(f'{label}: {getattr(exc, "stderr", None) or str(exc)}')
    if not shutil.which('bwrap'):
        problems.append('bubblewrap is missing (bwrap >= 0.5 required)')
    for name, url in (
        ('policy', os.environ.get('POLICY_URL', 'http://127.0.0.1:8600/v1').rstrip('/') + '/models'),
        ('molmo', os.environ.get('MOLMO_URL', 'http://127.0.0.1:8122/v1').rstrip('/') + '/models'),
        ('sam3', os.environ.get('SAM3_URL', 'http://127.0.0.1:8114').rstrip('/') + '/openapi.json'),
    ):
        try:
            with urlopen(url, timeout=10) as response:
                value = json.load(response)
            if name == 'sam3':
                if not all(key in value.get('paths', {}) for key in ('/segment', '/segment_point')):
                    raise ValueError('expected SAM 3 segmentation routes are missing')
                report[name] = 'segmentation API reachable (inference not yet checked)'
            else:
                data = value.get('data', [])
                if not data:
                    raise ValueError('no served models')
                report[name] = data
                if name == 'policy' and os.environ.get('POLICY_MODEL', 'captain-4b') not in [m['id'] for m in data]:
                    raise ValueError('POLICY_MODEL does not match a served model name')
        except (OSError, ValueError, KeyError) as exc:
            problems.append(f'{name}: {exc}')
    print(json.dumps(dict(checks=report, problems=problems), indent=2), flush=True)
    if problems:
        raise ValueError('baseline not ready; resolve the items above before running')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'check', 'run', 'summary'])
    parser.add_argument('--checkout', type=Path, default=PROJECT / 'data/captain/upstream')
    parser.add_argument('--plan', type=Path, default=PROJECT / 'data/captain/task0_official')
    parser.add_argument('--suite', default='libero_10')
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--full', action='store_true', help='prepare all 240 official episodes')
    parser.add_argument('--out', type=Path, help='fresh output per model; required for run/summary')
    parser.add_argument('--model-source', help='actual HF ID/revision or local model path; recorded, not verified')
    parser.add_argument('--parallel', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true', help='show official commands without launching episodes')
    args = parser.parse_args()
    args.checkout, args.plan = args.checkout.resolve(), args.plan.resolve()
    if args.action == 'prepare':
        prepare(args)
        return
    manifest = plan_check(args)
    if args.action == 'check':
        readiness(args)
        return
    if args.out is None:
        parser.error('--out is required')
    args.out = args.out.resolve()
    harness = args.checkout / 'harness'
    if args.action == 'summary':
        call(sys.executable, harness / 'eval/summarize.py', args.plan / 'tasks.tsv', args.out)
        return
    if args.parallel < 1:
        parser.error('--parallel must be positive')
    if not args.model_source:
        parser.error('--model-source is required to record which weights are being evaluated')
    if args.out.exists():
        raise ValueError('use a fresh --out directory for each model/trial')
    checks = {} if args.dry_run else readiness(args)
    args.out.mkdir(parents=True)
    selected_env = {key: os.environ[key] for key in ('POLICY_URL', 'POLICY_MODEL', 'CAPTAIN_PYTHON',
                   'STUDENT_PYTHON', 'EGL_DEVICE', 'SAM3_URL', 'MOLMO_URL') if key in os.environ}
    # Freeze TEMP at the official evaluation setting; this wrapper does not change prompts or tool schemas.
    env = dict(os.environ, CAPTAIN_HARNESS=str(harness), TEMP='1.0')
    command = [sys.executable, str(harness / 'eval/run_jobs.py'), str(args.plan / 'tasks.tsv'),
               '--out', str(args.out), '--parallel', str(args.parallel)]
    if args.dry_run:
        command.append('--dry-run')
    (args.out / 'baseline_manifest.json').write_text(json.dumps(dict(
        **manifest, model_source=args.model_source, model_identity_verified=False,
        endpoints=checks, environment=selected_env, command=command, dry_run=args.dry_run,
        temperature=1.0, top_p=1.0), indent=2) + '\n')
    print(f'Official runner: {manifest["episodes"]} episodes; logs: {args.out / "logs"}', flush=True)
    call(*command, env=env)
    if not args.dry_run:
        call(sys.executable, harness / 'eval/summarize.py', args.plan / 'tasks.tsv', args.out)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        sys.exit(f'[captain] {exc}')
