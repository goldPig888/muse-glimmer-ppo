"""The paper's LiberoEmbodiment, running ``paper_worker.py`` so replies carry reward info.

Everything the model sees (observation, action space, docs) is the paper's.
``extra_reset`` lets a caller start from a saved demo state.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from inspect_robots_libero import embodiment as module


class RewardLiberoEmbodiment(module.LiberoEmbodiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_reset: dict[str, Any] = {}

    def _start_worker(self):
        original = module.WORKER
        try:
            module.WORKER = Path(__file__).with_name('paper_worker.py')
            super()._start_worker()
        finally:
            module.WORKER = original

    def _rpc(self, request):
        if request.get('op') == 'reset' and self.extra_reset:
            request = {**request, **self.extra_reset}
        return super()._rpc(request)

    @property
    def progress(self) -> dict:
        return (self.last or {}).get('progress', {})
