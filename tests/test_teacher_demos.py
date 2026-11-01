import hashlib
import json
from pathlib import Path
import runpy

from PIL import Image
import pytest


EXPORT = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/prepare_teacher_demos.py'))


def make_episode(root, state):
    folder = root / f'run{state}'
    inspect = folder / 'episode_00/inspect'
    transcript = inspect / 'transcripts/t.jsonl'
    wire = inspect / 'wire/id/scene/calls.jsonl'
    transcript.parent.mkdir(parents=True)
    wire.parent.mkdir(parents=True)
    blobs = wire.parent.parent / 'blobs'
    blobs.mkdir()
    (folder / 'protocol.json').write_text('{}')
    (inspect / 'result.json').write_text(json.dumps({'results': {'metrics': {'success_at_end': 1}}}))
    tool = dict(type='function', function=dict(name='move_to', arguments={'targets': {'z': .7}}))
    messages = [dict(role='system', content='Robot instructions'), dict(role='user', content='Goal: place can')]
    requests = []
    for turn in range(3):
        image = blobs / 'tmp.png'
        Image.new('RGB', (2, 2), (state, turn, 0)).save(image)
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        image.rename(blobs / f'{digest}.png')
        content = [dict(type='text', text=f'observation {turn}'),
                   dict(type='text', text='[image omitted: streamed camera frame]')]
        messages.append(dict(role='user', content=content))
        request = dict(input=[dict(role='user', content=[
            dict(type='input_text', text=f'observation {turn}'),
            dict(type='input_image', image_url=f'data:image/png;base64,$blob:{digest}')])],
            tools=[dict(type='function', name='move_to', parameters={'type': 'object'})])
        response = dict(output=[dict(type='function_call', name='move_to', arguments={'targets': {'z': .7}}),
                                dict(type='reasoning', encrypted_content='NEVER EXPORT THIS')])
        requests.append(dict(status=200, call=turn, request=request, response=response))
        messages.append(dict(role='assistant', content=None, tool_calls=[tool]))
        messages.append(dict(role='tool', content='error: invalid' if turn == 1 else 'executing move_to'))
    transcript.write_text(''.join(json.dumps(m) + '\n' for m in messages))
    wire.write_text(''.join(json.dumps(m) + '\n' for m in requests))
    return dict(model='openai/gpt-6-astra', task_id=0, init_state=state,
                episode=0, run=f'run{state}', success=True)


def test_teacher_export_pairs_original_pixels_excludes_future_and_groups_states(tmp_path):
    source = tmp_path / 'source'
    rows = [make_episode(source, state) for state in (0, 8)]
    results = tmp_path / 'results.json'
    results.write_text(json.dumps({'episodes': rows}))
    output = tmp_path / 'dataset'
    audit = EXPORT['prepare'](results, [source], output, {0}, {8}, horizon=1)
    examples = [json.loads(line) for line in (output / 'examples.jsonl').read_text().splitlines()]
    assert audit['examples'] == {'train': 2, 'validation': 2}
    assert audit['accepted_episodes'] == 2
    assert [r['call'] for r in examples] == [1, 3, 1, 3]
    assert examples[0]['split'] == 'train' and examples[2]['split'] == 'validation'
    assert examples[0]['messages'][-1]['content'][0]['text'] == 'observation 0'
    assert 'observation 1' not in json.dumps(examples[0])
    assert 'NEVER EXPORT THIS' not in (output / 'examples.jsonl').read_text()
    for row in examples:
        images = [p['image'] for m in row['messages'] if isinstance(m.get('content'), list)
                  for p in m['content'] if p.get('type') == 'image']
        assert len(images) == 1
        with Image.open(output / images[0]) as image:
            assert image.getpixel((0, 0)) == (row['source']['init_state'], row['call'] - 1, 0)
    with pytest.raises(ValueError, match='fresh'):
        EXPORT['prepare'](results, [source], output, {0}, {8})
    with pytest.raises(ValueError, match='overlap'):
        EXPORT['prepare'](results, [source], tmp_path / 'overlap', {0}, {0})


def test_teacher_export_rejects_a_wrong_wire_response(tmp_path):
    source = tmp_path / 'source'
    row = make_episode(source, 0)
    wire = next(source.rglob('calls.jsonl'))
    events = [json.loads(line) for line in wire.read_text().splitlines()]
    events[0]['response']['output'][0]['arguments']['targets']['z'] = 1.
    wire.write_text(''.join(json.dumps(e) + '\n' for e in events))
    (tmp_path / 'images').mkdir()
    with pytest.raises(ValueError, match='differs from executed'):
        EXPORT['examples_from_episode'](source / row['run'], row, tmp_path, 2)
