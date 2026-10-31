"""Stress-test scoring/backprop on a recorded paper episode's full context.

The response is repeated to --response-tokens for memory testing; it is not
an on-policy training sample. No optimizer step is performed.
"""
import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from glimmer_ppo.paper_env import PaperEnv, PaperEnvConfig
from glimmer_ppo.policy import Policy, PolicyConfig
from glimmer_ppo.sim.embodiment import Start


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--response-tokens', type=int, default=2048)
    p.add_argument('--activation-offload', action='store_true')
    args = p.parse_args()
    protocol = json.loads((args.episode / 'protocol.json').read_text())
    events = [json.loads(line) for line in (args.episode / 'episode.jsonl').read_text().splitlines()]
    turns = [json.loads(line) for line in (args.episode / 'ppo_turns.jsonl').read_text().splitlines()]
    policy = Policy(PolicyConfig(load_in_4bit=True, activation_offload=args.activation_offload))
    env = PaperEnv(args.output / 'sim_check', PaperEnvConfig())
    try:
        env.reset(Start('libero_10', protocol['task_id'], protocol['init_state']))
        tools = env.tools
        messages = [dict(role='system', content=protocol['system_prompt'])]
        def observation(call, text):
            content = [dict(type='text', text=text)]
            for camera in ('agentview', 'wrist'):
                path = args.episode / 'frames' / f'call{call:02d}_{camera}.png'
                if path.exists():
                    with Image.open(path) as im:
                        image = im.convert('RGB')
                    content.extend([dict(type='text', text=f'{camera} camera'),
                                    dict(type='image', image=image)])
            return dict(role='user', content=content)
        messages.append(observation(0, events[0]['result']))
        by_call = {event['call']: event for event in events}
        for turn in turns:
            call = turn['parsed']
            messages.append(dict(role='assistant', content='' if call else turn['reply'],
                tool_calls=[dict(type='function', function=dict(name=call[0], arguments=call[1]))] if call else []))
            event = by_call[turn['decision']]
            messages.append(dict(role='tool', name=turn['tool'], content=event['result']))
            if (args.episode / 'frames' / f"call{turn['decision']:02d}_agentview.png").exists():
                messages.append(observation(turn['decision'], 'New observation.'))
        env.messages = messages
        env._strip_old_images()
        tokens = policy.processor.tokenizer.encode(turns[-1]['reply'], add_special_tokens=False)
        tokens = (tokens * (args.response_tokens // len(tokens) + 1))[:args.response_tokens]
        record = policy.record_for(env.messages, tools, tokens)
        print(json.dumps(dict(prompt_tokens=record['inputs']['input_ids'].shape[1],
                              response_tokens=len(tokens), recorded_decisions=len(turns))), flush=True)
        policy.eval()
        with torch.no_grad():
            logp, value = policy.score(record)
            ref, _ = policy.score(record, with_value=False, reference=True)
        assert torch.isfinite(logp).all() and torch.isfinite(ref).all() and torch.isfinite(value)
        del logp, ref, value
        print('Policy and reference scoring passed', flush=True)
        policy.train()
        logp, value = policy.score(record)
        (-logp.mean().to(value.device) + value.square()).backward()
        trainable = policy.trainable_parameters()
        assert any(p.grad is not None for p in trainable)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable)
        result = dict(passed=True, prompt_tokens=record['inputs']['input_ids'].shape[1],
                      response_tokens=len(tokens), peak_memory_gib=[
                          round(torch.cuda.max_memory_allocated(i)/2**30, 2)
                          for i in range(torch.cuda.device_count())])
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
