"""Where training episodes begin.

Two kinds of start, mixed by ``StartSampler``:

* ``init``: a LIBERO initial state (the task from the beginning).
* ``demo``: a simulator state saved from a human demo just before a grasp or
  just before a release (made by scripts/extract_start_states.py). These give
  short episodes with clear credit, the "mid-demo starts" of the plan.

LIBERO-10 requires explicit opt-in for single-task experiments.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from .sim.embodiment import Start

HELD_OUT_SUITES = {'libero_10'}


def parse_ids(text: str, upper: int) -> list[int]:
    ids: list[int] = []
    for part in text.split(','):
        part = part.strip()
        if '-' in part:
            a, b = map(int, part.split('-'))
            ids.extend(range(a, b + 1))
        elif part:
            ids.append(int(part))
    bad = [i for i in ids if not 0 <= i < upper]
    if bad:
        raise ValueError(f'ids out of range 0..{upper - 1}: {bad}')
    return ids


class StartSampler:
    def __init__(self, suite='libero_90', tasks=range(90), inits=range(50), demo_states: str | Path | None = None,
                 demo_fraction: float = 0.0, kinds=('pre_grasp', 'pre_release'), seed: int = 0,
                 allow_test_suite: bool = False):
        # The single-task experiment trains on LIBERO-10 task 0 itself, holding out
        # initial states instead of tasks; it must say so explicitly.
        if suite in HELD_OUT_SUITES and not allow_test_suite:
            raise ValueError(f'{suite} is the held-out test suite; pass allow_test_suite=True for a '
                             'single-task run that holds out initial states instead')
        if not 0.0 <= demo_fraction <= 1.0:
            raise ValueError('demo_fraction must be between 0 and 1')
        if not kinds or set(kinds) - {'pre_grasp', 'pre_release'}:
            raise ValueError('demo kinds must be pre_grasp and/or pre_release')
        self.suite, self.tasks, self.inits = suite, list(tasks), list(inits)
        self.rng = random.Random(seed)
        self.demo: list[Start] = []
        if demo_states:
            data = np.load(demo_states, allow_pickle=False)
            if 'suite' in data and str(data['suite'].item()) != suite:
                data.close()
                raise ValueError('Demo start suite does not match the requested suite')
            for i in range(len(data['task_id'])):
                if int(data['task_id'][i]) in self.tasks and str(data['kind'][i]) in kinds:
                    state = data['state'][i][:int(data['length'][i])]
                    kind = str(data['kind'][i])
                    self.demo.append(Start(self.suite, int(data['task_id'][i]), sim_state=state,
                                           settle_gripper=float(data['settle_gripper'][i]) if 'settle_gripper' in data
                                           else (1.0 if kind == 'pre_release' else -1.0),
                                           label=f"demo{int(data['demo'][i])}_{data['kind'][i]}_{int(data['step'][i])}"))
            data.close()
            if demo_fraction > 0 and not self.demo:
                raise ValueError('No demo starts match the requested tasks/kinds; refusing to fall back to initial states')
        self.demo_fraction = demo_fraction if self.demo else 0.0

    def __call__(self) -> Start:
        if self.demo and self.rng.random() < self.demo_fraction:
            return self.rng.choice(self.demo)
        task, init = self.rng.choice(self.tasks), self.rng.choice(self.inits)
        return Start(self.suite, task, init, label=f'init{init:02d}')
