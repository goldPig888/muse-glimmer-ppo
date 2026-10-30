"""Check llama.cpp rollouts against the Hugging Face training model, before training.

    CUDA_VISIBLE_DEVICES=1 python scripts/check_rollout_server.py \
        --llama-server ~/llama.cpp/build/bin/llama-server \
        --gguf /metadisk/may/models/Muse-Glimmer-30B-Q8_0.gguf \
        --mmproj /metadisk/may/models/mmproj-Muse-Glimmer-30B-F16.gguf \
        --server-gpu 0 --load-in-4bit

Renders the first task-0 prompt with the HF chat template, samples a few
replies on llama-server, then scores the returned token ids with the HF
model. It prints, per sample: tokens, whether the tool call parsed, and the
mean log-prob under HF. Healthy: calls parse, and the mean log-prob is not far
below what HF gives its own samples (run with --compare-hf to see that too).
A very low HF log-prob means the prompt or images differ between the two.
"""
import argparse
import json
from pathlib import Path

import torch

from glimmer_ppo.paper_env import PaperEnv, PaperEnvConfig
from glimmer_ppo.policy import Policy, PolicyConfig
from glimmer_ppo.rollout_server import LlamaServer, ServerConfig, generate_with_server
from glimmer_ppo.sim.embodiment import Start
from glimmer_ppo.toolcall import parse_tool_call


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--llama-server', required=True)
    p.add_argument('--gguf', required=True)
    p.add_argument('--mmproj', required=True)
    p.add_argument('--server-gpu', default='0')
    p.add_argument('--server-port', type=int, default=8090)
    p.add_argument('--model', default=PolicyConfig.model)
    p.add_argument('--load-in-4bit', action='store_true')
    p.add_argument('--samples', type=int, default=3)
    p.add_argument('--max-new-tokens', type=int, default=1024)
    p.add_argument('--reasoning', default='high')
    p.add_argument('--compare-hf', action='store_true', help='also sample with HF generate and score it')
    p.add_argument('--output', type=Path, default=Path('runs/check_rollout_server'))
    args = p.parse_args()

    policy = Policy(PolicyConfig(model=args.model, load_in_4bit=args.load_in_4bit,
                                 max_new_tokens=args.max_new_tokens, reasoning_strength=args.reasoning or None))
    policy.eval()
    server = LlamaServer(ServerConfig(args.llama_server, args.gguf, args.mmproj, gpu=args.server_gpu,
                                      port=args.server_port), args.output / 'server').start()
    env = PaperEnv(args.output / 'episodes', PaperEnvConfig())
    try:
        messages = env.reset(Start('libero_10', 0, 0))
        prompt, images = policy.render_prompt(messages, env.tools)
        print(f'prompt: {len(prompt)} chars, {len(images)} images, image token {policy.image_token!r}')
        rows = []
        for i in range(args.samples):
            text, truncated, rec = generate_with_server(server, policy, messages, env.tools)
            with torch.no_grad():
                logp, value = policy.score(rec)
            rows.append(dict(source='llama.cpp', tokens=len(rec['response']), truncated=truncated,
                             parsed=parse_tool_call(text), mean_logp=round(float(logp.mean()), 3),
                             value=round(float(value), 4)))
            print(json.dumps(rows[-1], default=str))
            (args.output / f'reply_{i}.txt').write_text(text)
        if args.compare_hf:
            for i in range(args.samples):
                text, truncated, rec = policy.act(messages, env.tools)
                with torch.no_grad():
                    logp, _ = policy.score(rec)
                print(json.dumps(dict(source='hf', tokens=len(rec['response']), truncated=truncated,
                                      parsed=parse_tool_call(text), mean_logp=round(float(logp.mean()), 3)),
                                 default=str))
        print(f'replies saved in {args.output}; send one if parsed is null')
    finally:
        env.close()
        server.stop()


if __name__ == '__main__':
    main()
