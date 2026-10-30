"""Check the paper-protocol env and reward on LIBERO-10 task 0, no model needed.

    python scripts/check_paper_env.py --task 0 --init 0

Plays four fixed replies in the formats the parser accepts: a move_to up
5 cm, a move_to below the z bound (invalid), the same invalid call again
(repeat penalty), and a reply with no tool call.
"""
import argparse
import json
from pathlib import Path

from glimmer_ppo.paper_env import PaperEnv, PaperEnvConfig
from glimmer_ppo.sim.embodiment import Start


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--init', type=int, default=0)
    p.add_argument('--output', type=Path, default=Path('runs/check_paper_env'))
    args = p.parse_args()
    env = PaperEnv(args.output, PaperEnvConfig(max_calls=4))
    try:
        messages = env.reset(Start('libero_10', args.task, args.init))
        print('system prompt chars:', len(messages[0]['content']))
        print('tools:', [t['function']['name'] for t in env.tools])
        print('goals:', env.progress['goals'], '->', env.progress['satisfied'])
        z = float(env.embodiment.last['eef_state'][2])
        bad = '{"name": "move_to", "arguments": {"targets": {"z": 0.1}, "note": "too low"}}'
        replies = [
            '<tool_call>{"name": "move_to", "arguments": {"targets": {"z": %.3f}, "note": "up 5 cm"}}</tool_call>' % (z + 0.05),
            bad,
            bad,
            'I am not sure what to do.',
        ]
        for reply in replies:
            reward, done, info = env.step(reply)
            print(json.dumps(dict(reward=round(reward, 4), done=done, status=info['status'], tool=info['tool'],
                                  steps=info['steps'], parts={k: round(v, 4) for k, v in info['reward_parts'].items()})))
            print('   tool result:', env.messages[-1 if done else -2]['content'].splitlines()[0][:110])
            if done:
                break
    finally:
        env.close()


if __name__ == '__main__':
    main()
