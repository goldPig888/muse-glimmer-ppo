"""Talks to ``worker.py`` over stdin/stdout; any LIBERO suite, any start state.

Implements only what ``llm_agent_memory.harness.session.Session`` uses from
its robot: ``reset``, ``step`` (returns ``terminated``/``termination_reason``),
``_rpc`` (for the motion checker), ``last``, ``episode_dir`` and ``close``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

WORKER = Path(__file__).with_name('worker.py')
ENV_SEED_BASE = 7000


@dataclass(frozen=True)
class Start:
    """Where an episode begins: a suite task plus an init state or a saved sim state."""
    suite: str
    task_id: int
    init_state: int = 0
    sim_state: Sequence[float] | None = None
    label: str = ''
    settle_gripper: float = -1.0   # +1 keeps the hand closed while settling (start holding an object)


@dataclass
class StepOutcome:
    terminated: bool
    termination_reason: str | None
    info: dict


class WorkerError(RuntimeError):
    pass


class SuiteEmbodiment:
    def __init__(self, output_dir, libero_python: str | None = None, record_video: bool = False):
        self.output = Path(output_dir)
        self.libero_python = libero_python or os.environ.get('LIBERO_PYTHON') or sys.executable
        self.record_video = record_video
        self.process: subprocess.Popen | None = None
        self.log = None
        self.last: dict[str, Any] | None = None
        self.episode_dir: Path | None = None
        self.instruction = ''
        self.task_name = ''

    def _start(self):
        self.output.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if not any(s in k for s in ('API_KEY', 'TOKEN', 'SECRET'))}
        env.update(MUJOCO_GL=env.get('MUJOCO_GL', 'egl'), PYOPENGL_PLATFORM=env.get('PYOPENGL_PLATFORM', 'egl'),
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   NUMBA_CACHE_DIR=env.get('NUMBA_CACHE_DIR', str(Path(tempfile.gettempdir()) / 'glimmer_ppo_numba')))
        self.log = (self.output / 'worker.log').open('a')
        self.process = subprocess.Popen([self.libero_python, '-u', str(WORKER)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.log, text=True, env=env)

    def _rpc(self, request: dict) -> dict:
        assert self.process and self.process.stdin and self.process.stdout
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise WorkerError(f'LIBERO worker exited; see {self.output / "worker.log"}')
        result = json.loads(line)
        if 'error' in result:
            raise WorkerError('LIBERO worker: ' + result['error'])
        if request['op'] in ('reset', 'step'):
            self.last = result
        return result

    def reset(self, start: Start):
        if self.process is None:
            self._start()
        stamp = time.strftime('%Y%m%d-%H%M%S')
        self.episode_dir = self.output / f'{stamp}_{start.suite}_task{start.task_id:02d}_{start.label or start.init_state}'
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        request = dict(op='reset', suite=start.suite, task_id=int(start.task_id), init_state=int(start.init_state),
                       env_seed=ENV_SEED_BASE + int(start.init_state),
                       sim_state=None if start.sim_state is None else np.asarray(start.sim_state, float).tolist(),
                       settle_gripper=float(start.settle_gripper),
                       video=str(self.episode_dir / 'rollout.mp4') if self.record_video else None)
        data = self._rpc(request)
        self.instruction = data['instruction']
        self.task_name = data.get('task_name', '')
        return data

    def step(self, action) -> StepOutcome:
        data = self._rpc({'op': 'step', 'action': np.asarray(getattr(action, 'data', action), float).tolist()})
        success = bool(data['success'])
        return StepOutcome(success, 'success' if success else None, {'success': success})

    def get_state(self) -> list[float]:
        return self._rpc({'op': 'get_state'})['sim_state']

    def close(self):
        if self.process is None:
            return
        try:
            self.process.stdin.write('{"op":"close"}\n')
            self.process.stdin.flush()
            self.process.wait(timeout=30)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            self.process.kill()
        finally:
            self.process = None
            if self.log:
                self.log.close()
                self.log = None
