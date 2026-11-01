"""Package successful Astra tool calls with their original pre-action camera images.

Reads an LLM_LIBERO experiment_results JSON (or .json.gz) and original Inspect
transcripts/wire captures. Makes no API calls and never replays a robot action.
The output is portable: image paths are relative to the dataset directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
from pathlib import Path
import re

from PIL import Image
from tqdm.auto import tqdm


def read_json(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        return json.load(stream)


def read_lines(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def ids(value):
    result = set()
    for item in value.split(','):
        ends = item.split('-')
        result.update(range(int(ends[0]), int(ends[-1]) + 1))
    return result


def canonical_call(item):
    function = item.get('function', item)
    args = function.get('arguments', {})
    if isinstance(args, str):
        args = json.loads(args)
    if not isinstance(args, dict):
        raise ValueError('Tool arguments must be an object')
    return dict(type='function', function=dict(name=function['name'], arguments=args))


def observation_key(content):
    return tuple(p['text'] for p in content if p.get('type') in ('text', 'input_text')
                 and not p.get('text', '').startswith('[image omitted'))


def copy_image(url, wire, output):
    match = re.fullmatch(r'data:image/png;base64,\$blob:([0-9a-f]{64})', url)
    if not match:
        raise ValueError('Expected an original PNG blob reference, not a video or missing image')
    digest = match[1]
    source = wire.parent.parent / 'blobs' / f'{digest}.png'
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError(f'Image checksum mismatch: {source}')
    with Image.open(source) as im:
        im.verify()
    relative = f'images/{digest}.png'
    destination = output / relative
    if not destination.exists():
        destination.write_bytes(payload)
    return dict(type='image', image=relative)


def wire_evidence(wire, output):
    observations, calls = {}, {}
    tools = None
    for row in read_lines(wire):
        if row.get('status') != 200:
            continue
        request = row['request']
        # Only visible transcripts and camera blobs are used. Reasoning items,
        # encrypted state, usage records and provider metadata are not exported.
        current_tools = [dict(type='function', function={
            k: t[k] for k in ('name', 'description', 'parameters') if k in t})
            for t in request.get('tools', []) if t.get('type') == 'function']
        if tools is not None and current_tools != tools:
            raise ValueError('Tool schema changed within a trajectory')
        tools = current_tools
        for m in request.get('input', []):
            content = m.get('content')
            if m.get('role') != 'user' or not isinstance(content, list):
                continue
            if not any(p.get('type') == 'input_image' for p in content):
                continue
            hydrated = []
            for part in content:
                if part.get('type') == 'input_text':
                    hydrated.append(dict(type='text', text=part['text']))
                elif part.get('type') == 'input_image':
                    hydrated.append(copy_image(part['image_url'], wire, output))
                else:
                    raise ValueError(f'Unsupported observation part: {part.get("type")}')
            key = observation_key(content)
            if key in observations and observations[key] != hydrated:
                raise ValueError('Same observation text maps to different images')
            observations[key] = hydrated
        response_calls = [canonical_call(item) for item in row.get('response', {}).get('output', [])
                          if item.get('type') == 'function_call']
        calls.setdefault(row['call'], []).append(response_calls)
    if not tools:
        raise ValueError('No original tool schemas')
    return observations, calls, tools


def examples_from_episode(folder, row, output, horizon):
    episode = folder / f'episode_{row["episode"]:02d}' / 'inspect'
    transcripts = list((episode / 'transcripts').rglob('*.jsonl'))
    wires = list((episode / 'wire').rglob('calls.jsonl'))
    if len(transcripts) != 1 or len(wires) != 1:
        raise ValueError('Expected one complete transcript and wire capture; continuation needs stitching')
    protocol = read_json(folder / 'protocol.json')
    if protocol.get('resume_from'):
        raise ValueError('Continuation needs stitching before training')
    # Confirm the archived outcome against the original simulator evidence.
    reports = [read_json(p) for p in episode.glob('*.json')]
    if not any(r.get('results', {}).get('metrics', {}).get('success_at_end') == 1 for r in reports):
        raise ValueError('Original simulator report does not confirm success')
    observations, wire_calls, tools = wire_evidence(wires[0], output)
    transcript = read_lines(transcripts[0])
    history, examples, excluded = [], [], Counter()
    call_index = 0
    for index, original in enumerate(transcript):
        role = original['role']
        content = original.get('content')
        if role == 'user' and isinstance(content, list):
            key = observation_key(content)
            if key not in observations:
                raise ValueError('Observation has no exact original camera match')
            message = dict(role='user', content=observations[key])
        elif role == 'assistant':
            calls = [canonical_call(t) for t in original.get('tool_calls', [])]
            message = dict(role='assistant', content=content or '', tool_calls=calls)
            # The executed transcript is authoritative; also require the saved
            # provider response to agree, protecting against retries/misalignment.
            if calls not in wire_calls.get(call_index, []):
                raise ValueError(f'Wire response differs from executed call {call_index + 1}')
            following = transcript[index + 1] if index + 1 < len(transcript) else {}
            valid = (len(calls) == 1 and calls[0]['function']['name'] == 'move_to'
                     and following.get('role') == 'tool'
                     and not str(following.get('content', '')).lower().startswith('error'))
            if valid:
                prompt = copy.deepcopy(history)
                image_users = [m for m in prompt if m['role'] == 'user' and isinstance(m['content'], list)
                               and any(p['type'] == 'image' for p in m['content'])]
                for old in image_users[:-horizon]:
                    old['content'] = [p if p['type'] != 'image' else dict(type='text', text='[image omitted]')
                                      for p in old['content']]
                examples.append(dict(messages=prompt, target=message, tools=tools,
                                     call=call_index + 1))
            else:
                excluded['non_move_or_error'] += 1
            call_index += 1
        elif role == 'tool':
            previous = history[-1].get('tool_calls', []) if history else []
            if len(previous) != 1:
                raise ValueError('Ambiguous tool result')
            message = dict(role='tool', name=previous[0]['function']['name'], content=content)
        elif role in ('system', 'user') and isinstance(content, str):
            message = dict(role=role, content=content)
        else:
            raise ValueError(f'Unsupported transcript message: {role}')
        history.append(message)
    return examples, excluded


def prepare(results, campaigns, output, train_inits, validation_inits, task=0,
            model='openai/gpt-6-astra', horizon=2):
    if train_inits & validation_inits:
        raise ValueError('Training and validation initial states overlap')
    if not train_inits or not validation_inits or horizon < 1:
        raise ValueError('Both state splits and a positive image horizon are required')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'images').mkdir()
    rows = [r for r in read_json(results)['episodes'] if r.get('model') == model
            and r.get('task_id') == task and r.get('success') is True
            and r.get('init_state') in train_inits | validation_inits]
    audit = dict(model=model, task=task, candidate_episodes=len(rows), episodes=[], skipped=[],
                 train_inits=sorted(train_inits), validation_inits=sorted(validation_inits),
                 note='Successful-episode imitation data, not replay-validated expert labels. '
                      'Original visible prompts are preserved. No future observations or hidden reasoning. '
                      'States used here must not be reported as untouched benchmark evaluation states.')
    totals, seen = Counter(), set()
    with (output / 'examples.jsonl').open('w') as stream:
        for row in tqdm(rows, desc='Package teacher episodes', unit='episode'):
            identity = {k: row.get(k) for k in ('model', 'effort', 'task_id', 'init_state', 'episode', 'run')}
            roots = [Path(row['source_campaign'])] if row.get('source_campaign') else []
            roots += list(campaigns)
            folder = next((root / row['run'] for root in roots if (root / row['run']).is_dir()), None)
            try:
                if folder is None:
                    raise ValueError('Original run directory not found')
                resolved = str(folder.resolve()) + f':{row["episode"]}'
                if resolved in seen:
                    raise ValueError('Duplicate source episode')
                seen.add(resolved)
                examples, excluded = examples_from_episode(folder, row, output, horizon)
                if not examples:
                    raise ValueError('No eligible move_to examples')
            except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
                audit['skipped'].append(dict(source=identity, reason=str(exc)))
                continue
            split = 'train' if row['init_state'] in train_inits else 'validation'
            for example in examples:
                stream.write(json.dumps(dict(split=split, source=identity, **example)) + '\n')
            totals[split] += len(examples)
            audit['episodes'].append(dict(source=identity, split=split, examples=len(examples),
                                          excluded=dict(excluded)))
    # Remove blobs from rejected episodes, keeping the portable package small.
    used = set()
    for example in read_lines(output / 'examples.jsonl'):
        for m in example['messages']:
            if isinstance(m.get('content'), list):
                used.update(p['image'] for p in m['content'] if p.get('type') == 'image')
    for image in (output / 'images').glob('*.png'):
        if str(image.relative_to(output)) not in used:
            image.unlink()
    audit.update(examples=dict(totals), images=len(used), accepted_episodes=len(audit['episodes']))
    (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    if not totals['train'] or not totals['validation']:
        raise ValueError('A split has no usable examples; inspect audit.json before training')
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--campaigns', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-inits', required=True)
    p.add_argument('--validation-inits', required=True)
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--model', default='openai/gpt-6-astra')
    args = p.parse_args()
    audit = prepare(args.results, args.campaigns, args.output, ids(args.train_inits),
                    ids(args.validation_inits), args.task, args.model)
    print(json.dumps({k: v for k, v in audit.items() if k not in ('episodes', 'skipped')}, indent=2))
    print(f'Skipped {len(audit["skipped"])} episodes; details: {args.output / "audit.json"}')


if __name__ == '__main__':
    main()
