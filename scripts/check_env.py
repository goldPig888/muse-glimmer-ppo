"""Check the LIBERO-90 environment and reward without loading any model.

    python scripts/check_env.py --list                 # task ids and instructions
    python scripts/check_env.py --task 0 --init 0      # three scripted replies, print rewards

The scripted replies are fixed strings in the model's reply format: an
observation request, a hover above the table centre, and a malformed reply.
They exercise parsing, motion, the checker and every reward term.
"""
import argparse
import json
from pathlib import Path

from glimmer_ppo.env import EnvConfig, HarnessEnv
from glimmer_ppo.sim.embodiment import Start

REPLIES = [
    "Looking first.\n# Thought: observe.\ncurl -s http://127.0.0.1:8081/obs",
    "Hover over the table.\n# Thought: move up and over.\n"
    "curl -s -X POST http://127.0.0.1:8081/plan --data-binary @- <<'EOF'\n"
    "robot.goto(x=0.0, y=0.0, z=1.0, pitch_deg=90, yaw_deg=0)\nEOF",
    "This reply has no command.",
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--suite', default='libero_90')
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--init', type=int, default=0)
    p.add_argument('--output', type=Path, default=Path('runs/check_env'))
    p.add_argument('--list', action='store_true')
    args = p.parse_args()
    if args.list:
        from libero.libero import benchmark
        suite = benchmark.get_benchmark_dict()[args.suite]()
        for i in range(suite.n_tasks):
            print(i, suite.get_task(i).language)
        return
    env = HarnessEnv(args.output, EnvConfig(max_decisions=len(REPLIES), record_video=True))
    try:
        messages = env.reset(Start(args.suite, args.task, args.init))
        progress = env.session.last['progress']
        print('instruction:', env.instruction)
        print('goal predicates:', progress['goals'], '-> satisfied', progress['satisfied'])
        print('system prompt chars:', len(messages[0]['content']))
        for reply in REPLIES:
            reward, done, info = env.step(reply)
            print(json.dumps(dict(reward=round(reward, 4), done=done, status=info['status'], steps=info['steps'],
                                  parts=info['reward_parts'])))
            if done:
                break
    finally:
        env.close()
    print('episode folder:', env.folder)


if __name__ == '__main__':
    main()
