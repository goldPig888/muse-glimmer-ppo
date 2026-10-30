"""Where training episodes begin.

Two kinds of start, mixed by ``StartSampler``:

* ``init``: a LIBERO initial state (the task from the beginning).
* ``demo``: a simulator state saved from a human demo just before a grasp or
  just before a release (made by scripts/extract_start_states.py). These give
  short episodes with clear credit, the "mid-demo starts" of the plan.

LIBERO-10 is never sampled here; it is the held-out test suite.
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
        self.suite, self.tasks, self.inits = suite, list(tasks), list(inits)
        self.rng = random.Random(seed)
        self.demo: list[Start] = []
        if demo_states:
            data = np.load(demo_states, allow_pickle=False)
            for i in range(len(data['task_id'])):
                if int(data['task_id'][i]) in self.tasks and str(data['kind'][i]) in kinds:
                    state = data['state'][i][:int(data['length'][i])]
                    kind = str(data['kind'][i])
                    self.demo.append(Start(self.suite, int(data['task_id'][i]), sim_state=state,
                                           settle_gripper=1.0 if kind == 'pre_release' else -1.0,
                                           label=f"demo{int(data['demo'][i])}_{data['kind'][i]}_{int(data['step'][i])}"))
        self.demo_fraction = demo_fraction if self.demo else 0.0

    def __call__(self) -> Start:
        if self.demo and self.rng.random() < self.demo_fraction:
            return self.rng.choice(self.demo)
        task, init = self.rng.choice(self.tasks), self.rng.choice(self.inits)
        return Start(self.suite, task, init, label=f'init{init:02d}')
