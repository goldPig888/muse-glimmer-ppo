#!/usr/bin/env python3
"""Check Glimmer's native image/tool template, then optionally a live Python loop.

The live check executes model-generated Python only in the pinned upstream
bubblewrap sandbox. This is an interface check, not robot success evaluation.
"""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import sys
from urllib.request import Request, urlopen

from PIL import Image

from glimmer_ppo.captain_bridge import PYTHON_TOOL, completion, normalize_request


def image_message():
    stream = io.BytesIO()
    Image.new('RGB', (32, 32), 'red').save(stream, format='PNG')
    return dict(role='user', content=[dict(type='text', text=(
        'Interface check only; no robot actions. This image is a red square. '
        'Use the python tool to set bridge_value = 41 and print(bridge_value). '
        'The same Python session will be used in the next request.')),
        dict(type='image_url', image_url=dict(url='data:image/png;base64,' +
             base64.b64encode(stream.getvalue()).decode()))])


def check_template(model):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    code = 'bridge_value = 41\nif bridge_value < 42:\n    print("a & b", bridge_value)\n'
    raw = (' to=python<|message|><atem:function_calls><atem:invoke name="python">'
           '<atem:parameter name="input">' + code + '</atem:parameter>'
           '</atem:invoke></atem:function_calls><|eot|>')
    assistant = completion(raw)['choices'][0]['message']
    messages = [image_message(), assistant,
                dict(role='tool', tool_call_id=assistant['tool_calls'][0]['id'], content='a & b 41'),
                dict(role='user', content='Print bridge_value + 1 using python.')]
    native, tools, images = normalize_request(dict(messages=messages, tools=[PYTHON_TOOL]))
    rendered = processor.apply_chat_template(native, tools=tools, tokenize=False,
                                             add_generation_prompt=True, reasoning_strength='high')
    assert code in rendered and '<tool_output name="python">' in rendered
    assert 'a & b 41' in rendered and images == 1
    encoded = processor.apply_chat_template(native, tools=tools, tokenize=True,
        return_dict=True, return_tensors='pt', add_generation_prompt=True, reasoning_strength='high')
    print(f'Native template passed: image + Python code + matched receipt; {encoded["input_ids"].shape[1]} prompt tokens', flush=True)


def live_check(args):
    from captain_baseline import verify
    verify(args.harness.parent)
    sys.path.insert(0, str(args.harness / 'src'))
    from captain_harness.react.python_repl import PythonRepl
    from captain_harness.react.sandbox import Sandbox
    from captain_harness.react.unified_exec import child_environment
    if not args.student_python:
        raise ValueError('set --student-python or STUDENT_PYTHON')
    if args.output.exists():
        raise ValueError('--output must be fresh')
    workspace = args.output / 'workspace'
    workspace.mkdir(parents=True)
    sandbox = Sandbox.detect(workspace, hidden_dirs=[args.output])
    if not sandbox.bwrap:
        raise ValueError('bubblewrap is required; no unsandboxed execution fallback')
    repl = PythonRepl(sandbox=sandbox, env=child_environment(os.environ), python=args.student_python)
    messages = [image_message()]
    rows = []
    try:
        for turn, expected in enumerate(('41', '42'), start=1):
            payload = dict(model=args.served_name, messages=messages, tools=[PYTHON_TOOL],
                           temperature=1.0, top_p=1.0, max_tokens=4096, stream=False)
            post = Request(args.endpoint.rstrip('/') + '/chat/completions',
                           json.dumps(payload).encode(), {'Content-Type': 'application/json'})
            with urlopen(post, timeout=900) as response:
                result = json.load(response)
            answer = result['choices'][0]['message']
            calls = answer.get('tool_calls') or []
            if result['choices'][0]['finish_reason'] == 'length' or not calls:
                raise ValueError('Glimmer returned no complete python call; inspect server requests.jsonl')
            messages.append(answer)
            output = ''
            for call in calls:
                function = call['function']
                if function['name'] != 'python':
                    raise ValueError('unsupported tool')
                code = json.loads(function['arguments'])['input']
                cell = repl.run(code, timeout_s=20)
                rows.append(dict(turn=turn, code=code, status=cell.status, output=cell.output,
                                 exception=cell.exception, generation=cell.generation))
                if cell.status != 'ok':
                    raise ValueError(f'Python cell failed: {cell.exception}')
                output += cell.output
                messages.append(dict(role='tool', tool_call_id=call['id'], content=cell.output))
            if expected not in output.split():
                raise ValueError(f'turn {turn}: expected printed {expected}, got {output!r}')
            print(f'Turn {turn}: {len(calls)} Python calls, printed {expected}; session generation={repl.generation}', flush=True)
            messages.append(dict(role='user', content='Using the existing bridge_value variable, print(bridge_value + 1) with python.'))
        if repl.restarts:
            raise ValueError('Python restarted; persistence check failed')
        print('Live interface passed. This does not measure robot performance.', flush=True)
    finally:
        (args.output / 'smoke.json').write_text(json.dumps(rows, indent=2) + '\n')
        repl.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='meta-models/Muse-Glimmer-30B')
    p.add_argument('--template-only', action='store_true')
    p.add_argument('--endpoint', default='http://127.0.0.1:8600/v1')
    p.add_argument('--served-name', default='glimmer')
    p.add_argument('--harness', type=Path, default=Path(__file__).resolve().parents[1] / 'data/captain/upstream/harness')
    p.add_argument('--student-python', default=os.environ.get('STUDENT_PYTHON'))
    p.add_argument('--output', type=Path, default=Path('runs/glimmer_python_smoke'))
    args = p.parse_args()
    args.harness, args.output = args.harness.resolve(), args.output.resolve()
    check_template(args.model)
    if not args.template_only:
        live_check(args)


if __name__ == '__main__':
    main()
