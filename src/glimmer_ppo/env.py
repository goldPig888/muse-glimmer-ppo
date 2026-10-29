"""The harness as an RL environment: reset(start) -> messages, step(reply) -> reward.

One step = one model decision (one reply, at most one /plan program), exactly
as in the zero-shot harness runs. The chat history follows the paper's
protocol: full text history, images only from the latest ``image_horizon``
observations.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from llm_agent_memory.harness.experiments import canonical
from llm_agent_memory.harness.session import Session
from llm_agent_memory.harness.upstream import extract_curl

from .prompt import system_prompt
from .reward import RewardConfig, decision_reward, status_of
from .sim.embodiment import Start, SuiteEmbodiment


@dataclass
class EnvConfig:
    max_decisions: int = 80
    max_steps: int = 900
    image_horizon: int = 2
    clean_images: bool = True
    record_video: bool = False
    reward: RewardConfig = field(default_factory=RewardConfig)


class PPOSession(Session):
    """The harness Session, driving any LIBERO suite through our worker."""

    def __init__(self, folder, cfg: EnvConfig, libero_python=None):
        super().__init__(folder, cfg.max_steps, compact_observations=True, efficient_contact=True,
                         clean_images=cfg.clean_images)
        self.robot = SuiteEmbodiment(self.folder, libero_python, record_video=cfg.record_video)


class HarnessEnv:
    def __init__(self, folder, cfg: EnvConfig | None = None, libero_python=None):
        self.folder = Path(folder)
        self.cfg = cfg or EnvConfig()
        self.libero_python = libero_python
        self.system = system_prompt(self.cfg.max_decisions, self.cfg.max_steps, self.cfg.clean_images)
        self.session: PPOSession | None = None
        self.episode = 0

    # -- episode control -------------------------------------------------
    def reset(self, start: Start) -> list[dict]:
        if self.session is not None:
            self.session.close()
        folder = self.folder / f'episode_{self.episode:05d}'
        folder.mkdir(parents=True, exist_ok=True)
        self.episode += 1
        self.session = PPOSession(folder, self.cfg, self.libero_python)
        self.session.reset(start)
        self.session._update_holding()   # a mid-demo start may begin holding an object
        self.start = start
        self.instruction = self.session.robot.instruction
        self.decisions = 0
        self.fraction = float(self.session.last['progress']['fraction'])
        self.grasped_once = bool(self.session.last['progress']['grasped_objects_of_interest'])
        self.failed = set()          # (canonical program, rounded pose) that did not complete
        self.messages = [dict(role='system', content=self.system)]
        self._observe()
        return self.messages

    def _observe(self):
        state, images = self.session.observe(self.decisions + 1)
        content = [dict(type='text', text='Task: ' + self.instruction + '\nObservation:\n' + json.dumps(state))]
        for name, png in zip(['agentview', 'wrist'], images):
            content += [dict(type='text', text=name + ' camera'),
                        dict(type='image', image=Image.open(io.BytesIO(png)).convert('RGB'))]
        self.messages.append(dict(role='user', content=content))
        users = [i for i, m in enumerate(self.messages) if m['role'] == 'user']
        for i in users[:-self.cfg.image_horizon]:
            self.messages[i]['content'] = [p for p in self.messages[i]['content'] if p['type'] != 'image']

    def step(self, reply: str, truncated: bool = False):
        """Run one model reply. Returns (reward, done, info)."""
        s = self.session
        self.decisions += 1
        pose_key = tuple(np.round(np.asarray(s.last['eef_state'][:3]), 2))
        program = None
        if truncated:
            s.feedback = 'Response truncated; nothing executed. Return a shorter complete command.'
        else:
            try:
                endpoint, program = extract_curl(reply)
                if endpoint == 'obs':
                    s.request_observation()
                else:
                    s.execute(program)
            except (ValueError, ArithmeticError) as exc:
                s.feedback = 'plan_error: ' + str(exc)
        status = status_of(s.feedback)
        key = None
        if program is not None:
            try:
                key = (canonical(program), pose_key)
            except Exception:
                key = (program.strip(), pose_key)
        repeated = key is not None and key in self.failed
        if key is not None and status not in ('completed', 'success'):
            self.failed.add(key)

        progress = s.last['progress']
        fraction = float(progress['fraction'])
        grasped = bool(progress['grasped_objects_of_interest'])
        first_grasp = grasped and not self.grasped_once
        self.grasped_once = self.grasped_once or grasped
        r = decision_reward(self.cfg.reward, prev_fraction=self.fraction, fraction=fraction,
                            success=s.success, first_grasp=first_grasp, status=status, repeated=repeated)
        self.fraction = fraction

        self.messages.append(dict(role='assistant', content=reply))
        done = s.success or s.done or self.decisions >= self.cfg.max_decisions
        info = dict(status=status, success=s.success, steps=s.steps, decisions=self.decisions,
                    goal_fraction=fraction, grasped=progress['grasped_objects_of_interest'],
                    reward_parts=r.parts, feedback=s.feedback)
        with (s.folder / 'turns.jsonl').open('a') as log:
            log.write(json.dumps(dict(decision=self.decisions, reply=reply, reward=r.total, **info)) + '\n')
        if not done:
            self._observe()
        return r.total, done, info

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None
