"""Read-only experiment dashboard. No training dependencies required.

python scripts/serve_training_viewer.py --runs-root runs --port 8770
python scripts/serve_training_viewer.py --runs-root runs --list
"""
from __future__ import annotations

import argparse
import json
import math
import logging
import re
import mimetypes
import os
from pathlib import Path
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    return value


class RunReader:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.lock = threading.Lock()
        self.files = {}
        self.gpu_at = 0
        self.gpus = []
        self.optim_offset = 0
        self.optim_inode = None
        self.optim_pending = b''
        self.optim_update = None
        self.optim_points = {}
        self.processes = None

    def kind(self):
        names = {p.name for p in self.root.iterdir() if p.is_file()}
        config = read_json(self.root / 'config.json')
        if 'server_config.json' in names:
            return 'python_server'
        if 'baseline_manifest.json' in names:
            return 'captain_eval'
        if 'evaluation_config.json' in names:
            return 'evaluation'
        if config.get('mode') == 'sft':
            return 'sft'
        if config.get('checkpoints'):
            return 'comparison'
        if 'report.json' in names:
            return 'critic'
        if 'smoke.json' in names:
            return 'python_check'
        if 'results.json' in names:
            return 'assistance_check'
        if 'config.json' in names:
            return 'ppo'
        if 'metrics.jsonl' in names:
            try:
                with (self.root / 'metrics.jsonl').open() as stream:
                    first = json.loads(stream.readline())
            except (OSError, ValueError):
                first = {}
            return 'sft' if 'epoch' in first else 'ppo'
        if 'results.jsonl' in names or 'summary.json' in names:
            return 'evaluation'
        if any(p.name.startswith('episode_') for p in self.root.iterdir() if p.is_dir()):
            return 'environment_check'
        return 'unknown'

    def process(self):
        if self.processes is not None:
            return self.processes.match(self.root)
        return None

    def tail(self):
        for name in ('train.log', 'server.log', 'eval.log'):
            try:
                with (self.root / name).open('rb') as stream:
                    stream.seek(0, 2)
                    stream.seek(max(0, stream.tell() - 18000))
                    return stream.read().decode('utf-8', errors='replace')
            except OSError:
                pass
        return ''

    def other_snapshot(self, selected=None):
        """Present each saved experiment's own metrics without inventing PPO values."""
        with self.lock:
            kind = self.kind()
            config = read_json(self.root / 'config.json') or read_json(self.root / 'server_config.json') or read_json(self.root / 'evaluation_config.json')
            args = config.get('args', config)
            metrics = list(self.rows(self.root / 'metrics.jsonl'))
            rows = list(self.rows(self.root / 'results.jsonl'))
            progress = read_json(self.root / 'progress.json')
            summary = read_json(self.root / 'summary.json') or read_json(self.root / 'report.json')
            pid = self.process()
            evidence = [self.root / name for name in ('config.json', 'metrics.jsonl', 'results.jsonl',
                'summary.json', 'report.json', 'progress.json', 'server_config.json', 'requests.jsonl',
                'evaluation_config.json', 'baseline_manifest.json', 'smoke.json', 'results.json', 'train.log')]
            evidence = [p for p in evidence if p.is_file()]
            episode_results = []
            folders = sorted([p for pattern in ('episodes/episode_*', 'episode_*', 'checkpoint_*/episodes/episode_*')
                              for p in self.root.glob(pattern) if p.is_dir() and p.resolve().is_relative_to(self.root)])
            if kind in ('environment_check', 'assistance_check', 'evaluation', 'comparison'):
                for folder in folders:
                    p = folder / 'result.json'
                    # No paths through symlinks outside the selected run.
                    if not p.resolve().is_relative_to(self.root):
                        continue
                    result = read_json(p)
                    if result.get('over'):
                        episode_results.append(dict(episode=p.parent.relative_to(self.root).as_posix(), **result))
                        evidence.append(p)
                if not rows:
                    rows = episode_results
            if kind == 'captain_eval':
                for p in self.root.glob('**/episode_result.json'):
                    if p.resolve().is_relative_to(self.root):
                        result = read_json(p)
                        if result:
                            rows.append(dict(episode=p.parent.relative_to(self.root).as_posix(), **result))
                            evidence.append(p)
                summary = dict(completed=len([r for r in rows if r.get('termination_reason') != 'infrastructure_error']),
                               infrastructure_errors=sum(r.get('termination_reason') == 'infrastructure_error' for r in rows),
                               host_successes=sum(bool(r.get('host_final_success')) for r in rows),
                               strict_successes=sum(bool(r.get('host_final_success') and r.get('agent_finished')) for r in rows))
                config = read_json(self.root / 'baseline_manifest.json')
                args = config
            cards, series = [], []
            def card(label, value, caption=''):
                cards.append(dict(label=label, value=value, caption=caption))
            if kind == 'sft':
                last = metrics[-1] if metrics else {}
                card('SFT epoch', f'{last.get("epoch", 0)} / {args.get("epochs", "—")}')
                card('Validation NLL', last.get('validation_nll'), 'Lower is better; this is not robot success')
                card('Training NLL', last.get('training_nll'))
                card('Training examples', config.get('train_examples'))
                series = [dict(key=key, label=label, x='epoch', rows=metrics) for key, label in
                          (('training_nll', 'Training token NLL'), ('validation_nll', 'Validation token NLL'))]
                rows = metrics
            elif kind == 'python_server':
                requests = list(self.rows(self.root / 'requests.jsonl'))
                calls = [r.get('response', {}).get('choices', [{}])[0].get('message', {}).get('tool_calls', []) for r in requests]
                card('Saved replies', len(requests))
                card('Python calls', sum(len(c) for c in calls))
                card('Parse errors', sum(bool(r.get('response', {}).get('glimmer_bridge', {}).get('parse_errors')) for r in requests))
                card('Model mode', 'Frozen · zero-shot', 'No optimizer or reward updates')
                rows = [dict(request=i + 1,
                             finish=r.get('response', {}).get('choices', [{}])[0].get('finish_reason'),
                             python_calls=len(calls[i]), **r) for i, r in enumerate(requests)]
                usage = [dict(request=i + 1, **r.get('response', {}).get('usage', {}), seconds=r.get('seconds')) for i, r in enumerate(requests)]
                series = [dict(key=key, label=label, x='request', rows=usage) for key, label in
                          (('completion_tokens', 'Generated tokens per request'), ('seconds', 'Seconds per request'))]
            elif kind in ('python_check', 'assistance_check'):
                path = self.root / ('smoke.json' if kind == 'python_check' else 'results.json')
                loaded = read_json(path)
                rows = loaded if isinstance(loaded, list) else []
                card('Saved checks', len(rows))
                card('Kind', 'Interface check' if kind == 'python_check' else 'Assistance physics check')
                if kind == 'assistance_check':
                    for gain in (1, 0):
                        trial = next((r for r in rows if r.get('gain') == gain), {})
                        card(f'Helper strength {gain}', 'Success' if trial.get('success') else 'Failure' if trial else '—', 'Scripted/idle physics, not learned policy evaluation')
                else:
                    card('Successful cells', sum(r.get('status') == 'ok' for r in rows))
                    card('Robot evaluation', 'No', 'Tests the Python interface only')
            elif kind == 'critic':
                card('Kind', 'Critic diagnostic')
                card('Actor changed', 'No' if summary.get('actor_modified') is False else 'Not recorded')
                rows = metrics
            elif kind == 'unknown':
                card('Folder', 'Found')
                card('Recognized records', 'None')
                rows = []
            else:
                evaluated = [r for r in rows if r.get('termination_reason') != 'infrastructure_error']
                success = sum(bool(r.get('success', r.get('host_final_success', False))) for r in evaluated)
                card('Completed episodes', len(evaluated))
                card('Successful episodes', success)
                card('Success rate', f'{100 * success / len(evaluated):.1f}%' if evaluated else '—',
                     'Pooled across checkpoints' if kind == 'comparison' else 'Only saved completed episodes')
                card('Evaluation target', progress.get('total', config.get('episodes', '—')))
                # A cumulative evaluation curve is descriptive, not a learning curve.
                cumulative = [dict(episode=i + 1, success_rate=sum(bool(r.get('success', r.get('host_final_success', False))) for r in evaluated[:i + 1]) / (i + 1)) for i in range(len(evaluated))]
                series = [dict(key='success_rate', label='Cumulative evaluation success · not a learning curve', x='episode', rows=cumulative)]
            final = progress.get('phase') == 'complete' or kind in ('python_check', 'assistance_check', 'critic') and bool(rows or summary)
            if kind == 'evaluation' and summary.get('episodes') and len(rows) >= summary['episodes']:
                final = True
            if kind == 'captain_eval' and len(rows) >= config.get('episodes', float('inf')):
                final = True
            phase = 'complete' if final else 'active' if pid else 'not running'
            if progress.get('phase') == 'error':
                phase = 'error'
            activity = max((p.stat().st_mtime for p in evidence if p.exists()), default=None)
            started = min((p.stat().st_mtime for p in evidence if p.exists()), default=None)
            descriptions = {
                'sft': 'Supervised imitation of saved teacher actions. Loss measures token prediction; robot success requires evaluation.',
                'python_server': 'Frozen Glimmer serving the CaPtain Python tool. Server activity alone is not a robot experiment. Replies appear after generation finishes.',
                'evaluation': 'Frozen policy evaluation. Results count completed episodes; no learning is performed here.',
                'comparison': 'Frozen checkpoint comparison. Inspect per-checkpoint and per-start results in the saved summary.',
                'captain_eval': 'Glimmer evaluation in the CaPtain harness. Host and strict success are separate; infrastructure errors are excluded.',
                'assistance_check': 'Object-force physics check. Assisted or scripted success does not establish policy learning.',
                'python_check': 'Python interface smoke check; not a robot success test.',
                'environment_check': 'Environment/tool check. These files are not evidence of trained policy accuracy.',
                'critic': 'Standalone critic diagnostic. This is not an actor-training result.',
                'unknown': 'Folder found, but no supported experiment records are present. It may be an empty or incomplete attempt.',
            }
            checkpoints = [p.relative_to(self.root).as_posix() for pattern in ('checkpoints/update_*', 'epoch_*') for p in sorted(self.root.glob(pattern)) if p.is_dir()]
            chosen = next((p for p in folders if p.relative_to(self.root).as_posix() == selected), folders[-1] if folders else None)
            frames = {}
            if chosen:
                for camera in ('agentview', 'wrist'):
                    images = list((chosen / 'frames').glob(f'call*_{camera}.png'))
                    if images:
                        image = max(images, key=lambda p: int(p.stem.split('_')[0][4:]))
                        frames[camera] = dict(path=image.relative_to(self.root).as_posix(), modified=image.stat().st_mtime_ns)
            return finite(dict(now=time.time(), run=self.root.name, path=str(self.root), kind=kind,
                pid=pid, running=bool(pid), phase=phase, started=started, last_activity=activity,
                config=args, metrics=metrics, cards=cards, series=series, records=rows[-100:],
                record_count=len(rows), summary=summary, progress=progress,
                description=descriptions[kind], checkpoints=checkpoints,
                artifacts=[p.relative_to(self.root).as_posix() for p in evidence],
                episodes=[dict(id=p.relative_to(self.root).as_posix()) for p in folders],
                selected=chosen.relative_to(self.root).as_posix() if chosen else None,
                episode_turns=list(self.rows(chosen / 'ppo_turns.jsonl'))[-100:] if chosen else [], frames=frames,
                gpus=self.gpu(), log=self.tail()))

    def optimizer_progress(self):
        """Recover small backward-pass updates from existing tqdm logs incrementally."""
        path = self.root / 'train.log'
        try:
            stat = path.stat()
            if stat.st_ino != self.optim_inode or stat.st_size < self.optim_offset:
                self.optim_offset, self.optim_pending = 0, b''
                self.optim_update, self.optim_points = None, {}
                self.optim_inode = stat.st_ino
            with path.open('rb') as f:
                f.seek(self.optim_offset)
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    self.optim_offset += len(chunk)
                    parts = re.split(b'[\r\n]', self.optim_pending + chunk)
                    self.optim_pending = parts.pop()
                    for part in parts:
                        line = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', part.decode('utf-8', errors='replace'))
                        update = re.search(r'\[ppo\] update (\d+): optimizing', line)
                        if update:
                            self.optim_update = int(update.group(1))
                            self.optim_points = {}
                        point = re.search(r'PPO backward passes:.*?(\d+)/(\d+).*?value_loss=([-+\d.eE]+)', line)
                        if point:
                            done, total, loss = int(point[1]), int(point[2]), float(point[3])
                            self.optim_points[done] = dict(step=done, total=total, value_loss=loss)
        except OSError:
            pass
        points = list(self.optim_points.values())[-4096:]
        return dict(update=self.optim_update, points=points,
                    completed=points[-1]['step'] if points else 0,
                    total=points[-1]['total'] if points else 0)

    def rows(self, path):
        """Read complete appended JSONL records; retry an unfinished last line."""
        try:
            stat = path.stat()
        except OSError:
            return []
        state = self.files.get(path)
        if state is None or state['inode'] != stat.st_ino or stat.st_size < state['offset']:
            state = dict(inode=stat.st_ino, offset=0, rows=[])
            self.files[path] = state
        with path.open('rb') as f:
            f.seek(state['offset'])
            for line in f:
                if not line.endswith(b'\n'):
                    break
                state['offset'] += len(line)
                try:
                    state['rows'].append(json.loads(line))
                except ValueError:
                    continue
        return state['rows']

    def gpu(self):
        if time.monotonic() - self.gpu_at < 10:
            return self.gpus
        self.gpu_at = time.monotonic()
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                '--format=csv,noheader,nounits',
            ], capture_output=True, text=True, timeout=3, check=True)
            self.gpus = []
            for line in result.stdout.splitlines():
                index, name, used, total, util = [s.strip() for s in line.split(',')]
                self.gpus.append(dict(index=int(index), name=name, used=int(used),
                                      total=int(total), utilization=int(util)))
        except (OSError, ValueError, subprocess.SubprocessError):
            self.gpus = []
        return self.gpus

    def snapshot(self, selected=None):
        if self.kind() != 'ppo':
            return self.other_snapshot(selected)
        with self.lock:
            config = read_json(self.root / 'config.json')
            args = config.get('args', {})
            metrics = list(self.rows(self.root / 'metrics.jsonl'))
            summaries = list(self.rows(self.root / 'episodes.jsonl'))
            folders = sorted((self.root / 'episodes').glob('episode_*'))
            if not summaries:
                summaries = [r for p in folders if (r := read_json(p / 'result.json')).get('over')]
            turns = {p.name: self.rows(p / 'ppo_turns.jsonl') for p in folders}
            total = sum(len(rows) for rows in turns.values())
            chosen = next((p for p in folders if p.name == selected), folders[-1] if folders else None)
            recent = list(turns.get(chosen.name, [])) if chosen else []
            protocol = read_json(chosen / 'protocol.json') if chosen else {}
            events = list(self.rows(chosen / 'episode.jsonl')) if chosen else []
            pid, running = None, False
            try:
                pid = int((self.root / 'train.pid').read_text().strip())
                cmdline = Path(f'/proc/{pid}/cmdline').read_bytes()
                state = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
                running = b'train_ppo.py' in cmdline and state != 'Z'
            except (OSError, ValueError, IndexError):
                pass
            if self.processes is not None:
                pid = self.process()
                running = bool(pid)
            target = int(args.get('updates', 100))
            update = int(metrics[-1]['update']) if metrics else int(config.get('resume_from_update', 0))
            complete = update >= target
            batch_size = int(args.get('decisions_per_update', 64))
            collected = max(0, total - sum(int(m.get('decisions', 0)) for m in metrics))
            phase = ('complete' if complete else 'stopped' if not running else
                     'optimizing' if collected >= batch_size else
                     'collecting' if folders else 'loading model')
            frames = {}
            if chosen:
                for camera in ('agentview', 'wrist'):
                    paths = list((chosen / 'frames').glob(f'call*_{camera}.png'))
                    if paths:
                        image = max(paths, key=lambda p: int(p.stem.split('_')[0][4:]))
                        frames[camera] = dict(path=image.relative_to(self.root).as_posix(),
                                              modified=image.stat().st_mtime_ns)
            activity_files = [self.root / 'train.log', self.root / 'metrics.jsonl']
            activity_files += [p / 'ppo_turns.jsonl' for p in folders]
            activity = max((p.stat().st_mtime for p in activity_files if p.exists()), default=None)
            started = (self.root / 'config.json').stat().st_mtime if args else None
            try:
                with (self.root / 'train.log').open('rb') as f:
                    f.seek(0, 2)
                    f.seek(max(0, f.tell() - 18000))
                    log = f.read().decode('utf-8', errors='replace')
            except OSError:
                log = ''
            return finite(dict(
                kind='ppo',
                now=time.time(), run=self.root.name, path=str(self.root), pid=pid, running=running,
                phase=phase, started=started, last_activity=activity, config=args,
                update=update, target_updates=target, total_decisions=total,
                batch_collected=collected, batch_size=batch_size, metrics=metrics,
                episode_count=len(folders), completed_episodes=len(summaries),
                successes=sum(bool(r.get('success')) for r in summaries),
                episodes=[dict(id=p.name, decisions=len(turns[p.name])) for p in reversed(folders)],
                selected=chosen.name if chosen else None, protocol=protocol,
                turns=recent, events=events[-100:], frames=frames, gpus=self.gpu(), log=log,
                optimizer=self.optimizer_progress(),
            ))


class ProcessReader:
    """Match owned experiment processes by their real output argument, never a viewer."""
    def __init__(self):
        self.at, self.entries = 0, []

    def match(self, root):
        if time.monotonic() - self.at > 5:
            self.at, self.entries = time.monotonic(), []
            allowed = {'train_ppo.py', 'train_sft.py', 'eval_task.py', 'compare_demo_checkpoints.py',
                       'serve_glimmer_python.py', 'check_glimmer_python.py', 'run_jobs.py',
                       'react_episode.py', 'check_object_assistance.py', 'train_physical_critic.py'}
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit():
                    continue
                try:
                    if proc.stat().st_uid != os.getuid():
                        continue
                    argv = proc.joinpath('cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
                    if not any(Path(v).name in allowed for v in argv[:3]):
                        continue
                    if proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                        continue
                    for i, arg in enumerate(argv):
                        if arg in ('--output', '--out', '--output-dir') and i + 1 < len(argv):
                            value = argv[i + 1]
                        elif any(arg.startswith(k + '=') for k in ('--output', '--out', '--output-dir')):
                            value = arg.split('=', 1)[1]
                        else:
                            continue
                        path = Path(value)
                        path = path if path.is_absolute() else proc.joinpath('cwd').resolve() / path
                        self.entries.append((path.resolve(), int(proc.name)))
                except (OSError, ValueError, IndexError):
                    continue
        return next((pid for path, pid in self.entries if path == root or path.is_relative_to(root)), None)


class RunCatalog:
    def __init__(self, root, initial=None, training_only=True):
        self.root = Path(root).resolve()
        self.initial = Path(initial).resolve() if initial else None
        self.training_only = training_only
        self.readers = {}
        self.processes = ProcessReader()
        self.lock = threading.Lock()
        self.at, self.entries = 0, []

    def discover(self):
        with self.lock:
            if time.monotonic() - self.at < 5:
                return list(self.entries)
            markers = {'config.json', 'server_config.json', 'evaluation_config.json', 'metrics.jsonl',
                       'summary.json', 'baseline_manifest.json', 'report.json', 'smoke.json', 'results.json', 'results.jsonl'}
            entries = []
            for folder, dirs, files in os.walk(self.root, followlinks=False):
                path = Path(folder)
                depth = len(path.relative_to(self.root).parts)
                dirs[:] = [d for d in dirs if not d.startswith('.') and not (path / d).is_symlink()
                           and d not in ('frames', 'images', 'workspace', 'checkpoints', 'logs', 'driver')
                           and not d.startswith('epoch_')]
                if depth >= 4:
                    dirs[:] = []
                if depth != 1 and not set(files).intersection(markers) and not any(d.startswith('episode_') for d in dirs):
                    continue
                key = path.relative_to(self.root).as_posix()
                reader = self.readers.setdefault(key, RunReader(path))
                reader.processes = self.processes
                kind = reader.kind()
                if self.training_only and kind not in ('ppo', 'sft'):
                    dirs[:] = []
                    continue
                times = [(path / name).stat().st_mtime for name in files if name in markers]
                entries.append(dict(id=key, name=key, path=str(path), kind=kind,
                                    running=bool(reader.process()), last_activity=max(times, default=path.stat().st_mtime)))
                dirs[:] = []  # This run owns its episodes/checkpoints; don't list them as experiments.
            entries.sort(key=lambda e: (not e['running'], -e['last_activity'], e['id']))
            self.entries, self.at = entries, time.monotonic()
            return list(entries)

    def selected(self, key=None):
        entries = self.discover()
        if key is None and self.initial:
            key = self.initial.relative_to(self.root).as_posix()
        if key is None:
            key = entries[0]['id'] if entries else None
        if key not in {e['id'] for e in entries}:
            raise FileNotFoundError('Unknown experiment')
        return self.readers[key], key


def make_handler(reader):
    page = Path(__file__).resolve().parents[1] / 'viewer' / 'index.html'

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, body, content_type):
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            request = urlsplit(self.path)
            try:
                if request.path == '/':
                    self.send(page.read_bytes(), 'text/html; charset=utf-8')
                elif request.path == '/live.js':
                    self.send((page.parent / 'live.js').read_bytes(), 'text/javascript; charset=utf-8')
                elif request.path == '/experiments.js':
                    self.send((page.parent / 'experiments.js').read_bytes(), 'text/javascript; charset=utf-8')
                elif request.path == '/api/state':
                    query = parse_qs(request.query)
                    selected = query.get('episode', [None])[0]
                    active, key = reader.selected(query.get('run', [None])[0]) if isinstance(reader, RunCatalog) else (reader, None)
                    state = active.snapshot(selected)
                    state['run_id'] = key
                    state['experiment_note'] = read_json(page.parent / 'experiment_notes.json').get(active.root.name)
                    body = json.dumps(state, allow_nan=False).encode()
                    self.send(body, 'application/json')
                elif request.path == '/api/runs':
                    entries = reader.discover() if isinstance(reader, RunCatalog) else [dict(id='.', name=reader.root.name, kind=reader.kind())]
                    self.send(json.dumps(dict(runs=entries), allow_nan=False).encode(), 'application/json')
                elif request.path.startswith('/frames/'):
                    active, _ = reader.selected(parse_qs(request.query).get('run', [None])[0]) if isinstance(reader, RunCatalog) else (reader, None)
                    path = (active.root / unquote(request.path[len('/frames/'):])).resolve()
                    try:
                        path.relative_to(active.root)
                    except ValueError:
                        self.send_error(404)
                        return
                    if path.suffix != '.png' or path.parent.name != 'frames':
                        self.send_error(404)
                        return
                    self.send(path.read_bytes(), mimetypes.guess_type(str(path))[0] or 'image/png')
                else:
                    self.send_error(404)
            except FileNotFoundError:
                self.send_error(404)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                logging.exception('Viewer request failed: %s', exc)
                self.send_error(500, 'Could not read run data')

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, help='initial experiment (optional with --runs-root)')
    parser.add_argument('--runs-root', type=Path, help='discover experiment folders under this directory')
    parser.add_argument('--list', action='store_true', help='print discovered experiments as JSON and exit')
    parser.add_argument('--include-nontraining', action='store_true', help='also show evaluations and checks (hidden by default)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    if args.run is None and args.runs_root is None:
        parser.error('provide --runs-root runs or --run RUN')
    root = args.runs_root or args.run.resolve().parent
    if not root.is_dir():
        parser.error('runs root does not exist')
    if args.run and not args.run.resolve().is_relative_to(root.resolve()):
        parser.error('--run must be inside --runs-root')
    catalog = RunCatalog(root, args.run, training_only=not args.include_nontraining)
    if args.list:
        print(json.dumps(dict(root=str(root.resolve()), experiments=catalog.discover()), indent=2))
        return
    server = ThreadingHTTPServer((args.host, args.port), make_handler(catalog))
    print(f'Live experiment viewer: http://{args.host}:{args.port} · runs: {root.resolve()}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
