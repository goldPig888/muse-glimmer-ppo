"""The paper's protocol as an RL environment: Muse writes move_to / done / give_up directly.

Motion, budgets and success are the paper's Algorithm 1, reused unchanged
from ``llm_agent_memory.episode.Episode`` (the upstream agent Toolset
validates, bounds and interpolates each move_to). This module only adds:
the chat messages for a Hugging Face model, tool-call parsing, and the reward.

Messages: system prompt (the paper's, for this task) -> user (goal + first
observation + both camera images) -> for each decision: assistant tool call
-> tool result text -> user message with the new camera images. Images are
kept for the last ``image_horizon`` observations, as in the paper.
"""
from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from llm_agent_memory.episode import Episode
from llm_agent_memory.protocol import MAX_CALLS, MAX_STEPS, system_prompt

from .reward import PotentialConfig, potential_reward
from .sim.embodiment import Start
from .sim.paper_embodiment import RewardLiberoEmbodiment
from .toolcall import parse_tool_call


@dataclass
class PaperEnvConfig:
    max_calls: int = MAX_CALLS
    max_steps: int = MAX_STEPS
    image_horizon: int = 2
    reward: PotentialConfig = field(default_factory=PotentialConfig)


class PaperEnv:
    def __init__(self, folder, cfg: PaperEnvConfig | None = None):
        self.folder = Path(folder)
        self.cfg = cfg or PaperEnvConfig()
        self.episode: Episode | None = None
        self.count = 0

    @property
    def tools(self):
        return self.episode.tool_schemas() if self.episode else None

    def reset(self, start: Start) -> list[dict]:
        if start.suite != 'libero_10':
            raise ValueError('the paper-protocol env runs LIBERO-10 tasks (the paper\'s embodiment)')
        self.close()
        run_dir = self.folder / f'episode_{self.count:05d}_task{start.task_id:02d}_{start.label or start.init_state}'
        self.count += 1
        embodiments = []

        def factory(task_id, run_dir):
            e = RewardLiberoEmbodiment(task_id=task_id, output_dir=run_dir)
            if start.sim_state is not None:
                e.extra_reset = dict(sim_state=[float(v) for v in start.sim_state],
                                     settle_gripper=float(start.settle_gripper))
            if start.assistance is not None:
                e.extra_reset['object_assistance'] = start.assistance
            embodiments.append(e)
            return e

        self.episode = Episode(start.task_id, start.init_state, run_dir, factory=factory,
                               max_calls=self.cfg.max_calls, max_steps=self.cfg.max_steps,
                               meta=dict(start=start.label, ppo=True,
                                   start_kind='full_demo_initial_state' if start.assistance else
                                              'mid_demo' if start.sim_state is not None else 'benchmark_initial_state',
                                   assistance_gain=start.assistance['gain'] if start.assistance else 0.))
        self.embodiment = embodiments[0]
        reply = self.episode.start()
        self.start = start
        self.progress = self.embodiment.progress
        self.assistance_returns = dict(self.progress.get('voc', {}).get('returns', {}))
        self.failed_calls = set()
        self.messages = [dict(role='system', content=system_prompt(start.task_id, self.cfg.max_calls)),
                         self._user(reply.text, reply.frames)]
        self._strip_old_images()
        return self.messages

    @staticmethod
    def _user(text, frames):
        content = [dict(type='text', text=text)]
        for f in frames:
            content += [dict(type='text', text=f'{f.camera} camera'),
                        dict(type='image', image=Image.open(io.BytesIO(f.png)).convert('RGB'))]
        return dict(role='user', content=content)

    def _strip_old_images(self):
        users = [i for i, m in enumerate(self.messages)
                 if m['role'] == 'user' and isinstance(m['content'], list)
                 and any(p['type'] == 'image' for p in m['content'])]
        for i in users[:-self.cfg.image_horizon]:
            self.messages[i]['content'] = [p if p['type'] != 'image' else dict(type='text', text='[image omitted]')
                                           for p in self.messages[i]['content']]

    def step(self, reply: str, truncated: bool = False):
        """Run one model reply. Returns (reward, done, info)."""
        ep = self.episode
        call = None if truncated else parse_tool_call(reply)
        if call is None:
            name, args = 'invalid_reply', {}
        else:
            name, args = call
        key = (name, json.dumps(args, sort_keys=True))
        repeated = key in self.failed_calls
        result = ep.call(name, args)
        invalid = result.text.startswith('error:')
        if invalid:
            self.failed_calls.add(key)

        progress = self.embodiment.progress
        r = potential_reward(self.cfg.reward, prev_progress=self.progress, progress=progress,
                             success=ep.success, invalid=invalid, repeated=repeated)
        if 'voc' in progress:
            current = progress['voc']['returns']
            for key, value in current.items():
                r.parts[key] = float(value) - self.assistance_returns.get(key, 0.)
            self.assistance_returns = dict(current)
            r.total = sum(r.parts.values())
        self.progress = progress

        self.messages.append(dict(role='assistant', content=reply if call is None else '',
                                  tool_calls=[] if call is None else
                                  [dict(type='function', function=dict(name=name, arguments=args))]))
        self.messages.append(dict(role='tool', name=name, content=result.text))
        done = ep.over
        if not done and result.frames:
            self.messages.append(self._user('New observation.', result.frames))
            self._strip_old_images()
        info = dict(status='invalid' if invalid else (ep.outcome or 'ok'), tool=name, success=ep.success,
                    outcome=ep.outcome, steps=ep.steps, decisions=ep.calls,
                    goal_fraction=float(progress.get('fraction', 0.0)),
                    grasped=progress.get('grasped_objects_of_interest', []),
                    lifted=progress.get('lifted_objects_of_interest', []),
                    reach_m=progress.get('reach_m'), carry_m=progress.get('carry_m'),
                    reward_parts=r.parts)
        if 'voc' in progress:
            info['voc'] = progress['voc']
        with (ep.run_dir / 'ppo_turns.jsonl').open('a') as log:
            log.write(json.dumps(dict(decision=ep.calls, reply=reply, parsed=call, reward=r.total,
                                      **{k: v for k, v in info.items()}), default=str) + '\n')
        return r.total, done, info

    def close(self):
        if self.episode is not None:
            self.episode.close()
            self.episode = None
