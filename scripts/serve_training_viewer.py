"""Read-only live PPO dashboard. No training dependencies required.

python scripts/serve_training_viewer.py --run /metadisk/may/ppo/task0_both
"""
from __future__ import annotations

import argparse
import json
import math
import logging
import mimetypes
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
        with self.lock:
            config = read_json(self.root / 'config.json')
            args = config.get('args', {})
            metrics = list(self.rows(self.root / 'metrics.jsonl'))
            summaries = list(self.rows(self.root / 'episodes.jsonl'))
            folders = sorted((self.root / 'episodes').glob('episode_*'))
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
            target = int(args.get('updates', 100))
            update = int(metrics[-1]['update']) if metrics else 0
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
                now=time.time(), run=self.root.name, path=str(self.root), pid=pid, running=running,
                phase=phase, started=started, last_activity=activity, config=args,
                update=update, target_updates=target, total_decisions=total,
                batch_collected=collected, batch_size=batch_size, metrics=metrics,
                episode_count=len(folders), completed_episodes=len(summaries),
                successes=sum(bool(r.get('success')) for r in summaries),
                episodes=[dict(id=p.name, decisions=len(turns[p.name])) for p in reversed(folders)],
                selected=chosen.name if chosen else None, protocol=protocol,
                turns=recent, events=events[-100:], frames=frames, gpus=self.gpu(), log=log,
            ))


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
                elif request.path == '/api/state':
                    selected = parse_qs(request.query).get('episode', [None])[0]
                    body = json.dumps(reader.snapshot(selected), allow_nan=False).encode()
                    self.send(body, 'application/json')
                elif request.path.startswith('/frames/'):
                    path = (reader.root / unquote(request.path[len('/frames/'):])).resolve()
                    try:
                        path.relative_to(reader.root)
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
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    if not (args.run / 'config.json').is_file():
        parser.error('--run must contain config.json')
    server = ThreadingHTTPServer((args.host, args.port), make_handler(RunReader(args.run)))
    print(f'Live PPO viewer: http://{args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
