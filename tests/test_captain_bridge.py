import base64
import importlib.util
import io
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from PIL import Image
import pytest

from glimmer_ppo.captain_bridge import PYTHON_TOOL, completion, normalize_request, parse_reply


def native(code, *, prefix=' to=python<|message|>', end='<|eot|>'):
    return (prefix + '<atem:function_calls>\n<atem:invoke name="python">\n'
            '<atem:parameter name="input">' + code + '</atem:parameter>\n'
            '</atem:invoke>\n</atem:function_calls>' + end)


def request(messages=None):
    return dict(model='glimmer', tools=[PYTHON_TOOL], messages=messages or [dict(role='user', content='test')])


def image_part():
    stream = io.BytesIO()
    Image.new('RGB', (12, 8), 'red').save(stream, format='PNG')
    return dict(type='image_url', image_url=dict(url='data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode()))


def test_native_python_preserves_code_and_multiple_calls_ignores_thought():
    code = 'x = [v for v in range(5) if v < 3]\nif x:\n    print("a &amp; b > c", x)\n'
    thought = native('raise Exception("example only")', prefix=' to=self<|message|>', end='<|eom|>')
    raw = thought + native(code, prefix='<|start|>assistant to=python<|message|>', end='<|eom|>')
    raw += native('print(x)', prefix='<|start|>assistant to=python<|message|>')
    message, finish, errors = parse_reply(raw)
    assert finish == 'tool_calls' and not errors
    calls = message['tool_calls']
    assert len(calls) == 2 and calls[0]['id'] != calls[1]['id']
    assert json.loads(calls[0]['function']['arguments']) == {'input': code}
    assert json.loads(calls[1]['function']['arguments']) == {'input': 'print(x)'}
    assert 'example only' in message['reasoning_content']


def test_truncated_or_malformed_never_forwards_partial_code():
    for raw, truncated in ((native('print(1)'), True),
                           (native('print(1)') + '<|start|>assistant to=python', False),
                           (native('print(1)').replace('name="input"', 'name="wrong"'), False),
                           (native('print(1)').replace('name="python"', 'name="move_to"'), False)):
        message, finish, errors = parse_reply(raw, truncated=truncated)
        assert not message.get('tool_calls') and errors
        assert finish == ('length' if truncated else 'stop')
    message, finish, errors = parse_reply(' to=user<|message|>Finished.<|eot|><|end_of_text|>')
    assert message['content'] == 'Finished.' and finish == 'stop' and not errors


def test_images_receipts_ids_and_native_mapping_round_trip():
    answer = completion(native('saved = 41\nprint(saved)'))['choices'][0]['message']
    calls = answer['tool_calls']
    messages = [dict(role='system', content='test'),
                dict(role='user', content=[dict(type='text', text='look'), image_part()]),
                answer, dict(role='tool', tool_call_id=calls[0]['id'], content='41'),
                dict(role='user', content='reuse saved')]
    original = json.dumps(messages)
    converted, tools, count = normalize_request(request(messages))
    assert count == 1 and converted[1]['content'][1]['image'].getpixel((0, 0)) == (255, 0, 0)
    assert converted[2]['tool_calls'][0]['function']['arguments']['input'].startswith('saved = 41')
    assert converted[3]['name'] == 'python' and converted[3]['content'] == '41'
    assert json.dumps(messages) == original  # caller history remains OpenAI-compatible
    messages[3]['tool_call_id'] = 'unknown'
    with pytest.raises(ValueError, match='preceding call'):
        normalize_request(request(messages))


def test_invalid_images_and_missing_receipts_fail_explicitly():
    bad = image_part()
    bad['image_url']['url'] = 'file:///etc/passwd'
    with pytest.raises(ValueError, match='inline base64'):
        normalize_request(request([dict(role='user', content=[bad])]))
    assistant = completion(native('print(1)'))['choices'][0]['message']
    with pytest.raises(ValueError, match='without results'):
        normalize_request(request([assistant, dict(role='user', content='continue')]))


def test_http_bridge_roundtrip_with_fake_generation(tmp_path):
    """Real HTTP and native template-shaped replies; no model or simulator required."""
    spec = importlib.util.spec_from_file_location('serve_glimmer_python', Path(__file__).parents[1] / 'scripts/serve_glimmer_python.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Fake:
        args = SimpleNamespace(served_name='glimmer', model='fixture')

        def generate(self, body):
            if body.get('test_context_error'):
                raise ValueError('context_length_exceeded: fixture')
            messages, _, images = normalize_request(body)
            assert images == 1
            code = 'print(saved + 1)' if messages[-1]['content'] == 'continue' else 'saved = 41\nprint(saved)'
            raw = native(code)
            return completion(raw), raw

    server = module.make_server(Fake(), '127.0.0.1', 0, tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f'http://127.0.0.1:{server.server_address[1]}'
    try:
        with urlopen(endpoint + '/v1/models') as response:
            assert json.load(response)['data'][0]['id'] == 'glimmer'
        messages = [dict(role='user', content=[image_part()])]
        for index in range(2):
            post = Request(endpoint + '/v1/chat/completions', json.dumps(request(messages)).encode(),
                           {'Content-Type': 'application/json'})
            with urlopen(post) as response:
                answer = json.load(response)['choices'][0]['message']
            assert len(answer['tool_calls']) == 1
            if index == 0:
                call = answer['tool_calls'][0]
                messages += [answer, dict(role='tool', tool_call_id=call['id'], content='41'),
                             dict(role='user', content='continue')]
            else:
                assert json.loads(answer['tool_calls'][0]['function']['arguments'])['input'] == 'print(saved + 1)'
        assert len((tmp_path / 'requests.jsonl').read_text().splitlines()) == 2
        overflow = Request(endpoint + '/v1/chat/completions', json.dumps({'test_context_error': True}).encode(),
                           {'Content-Type': 'application/json'})
        with pytest.raises(HTTPError) as error:
            urlopen(overflow)
        assert error.value.code == 400
        assert json.load(error.value)['error']['code'] == 'context_length_exceeded'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
