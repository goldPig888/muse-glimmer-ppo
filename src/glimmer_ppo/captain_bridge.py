"""Frozen Glimmer ↔ CaPtain chat protocol. Python executes only in the harness.

Muse's ATEM template explicitly specifies raw string parameters, not XML. In
particular, Python containing <, >, &, quotes and indentation must round-trip.
"""
from __future__ import annotations

import base64
import copy
import io
import json
import re
import uuid

from PIL import Image


PYTHON_TOOL = dict(type='function', function=dict(
    name='python', description='Execute Python in the persistent harness session.',
    parameters=dict(type='object', properties=dict(input=dict(type='string')),
                    required=['input'], additionalProperties=False)))


def normalize_request(body):
    """Convert OpenAI images/JSON args into the native HF template representation."""
    if body.get('stream'):
        raise ValueError('streaming is not supported; use stream=false')
    if body.get('tool_choice', 'auto') != 'auto':
        raise ValueError('only tool_choice=auto is supported')
    if body.get('n', 1) != 1:
        raise ValueError('only n=1 is supported')
    tools = copy.deepcopy(body.get('tools') or [])
    if len(tools) != 1 or tools[0].get('type') != 'function' or tools[0].get('function', {}).get('name') != 'python':
        raise ValueError('this bridge requires the harness python function tool')
    schema = tools[0]['function'].get('parameters', {})
    if schema.get('properties', {}).get('input', {}).get('type') != 'string':
        raise ValueError('python tool must have a string input parameter')
    messages = copy.deepcopy(body.get('messages'))
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages must be a nonempty list')
    calls, results, images = {}, set(), 0
    for message in messages:
        if message.get('role') not in ('system', 'user', 'assistant', 'tool'):
            raise ValueError('unsupported message role')
        for call in message.get('tool_calls') or []:
            function = call['function']
            args = function.get('arguments', {})
            args = json.loads(args) if isinstance(args, str) else args
            if function.get('name') != 'python' or not isinstance(args, dict) or set(args) != {'input'} or not isinstance(args['input'], str):
                raise ValueError('history contains an invalid python call')
            call_id = call.get('id')
            if not call_id or call_id in calls:
                raise ValueError('tool call IDs must be present and unique')
            calls[call_id] = 'python'
            function['arguments'] = args
        if message['role'] == 'tool':
            call_id = message.get('tool_call_id')
            if call_id not in calls or call_id in results:
                raise ValueError('tool result has no unique preceding call')
            results.add(call_id)
            message['name'] = 'python'
        content = message.get('content')
        if content is None:
            message['content'] = ''
        elif isinstance(content, list):
            parts = []
            for part in content:
                if part.get('type') == 'text':
                    parts.append(dict(type='text', text=part['text']))
                elif part.get('type') == 'image_url':
                    url = part['image_url']
                    url = url.get('url') if isinstance(url, dict) else url
                    if not isinstance(url, str) or not re.match(r'^data:image/[\w.+-]+;base64,', url):
                        raise ValueError('images must be inline base64 data:image URLs')
                    raw = base64.b64decode(url.split(',', 1)[1], validate=True)
                    with Image.open(io.BytesIO(raw)) as image:
                        image.load()
                        parts.append(dict(type='image', image=image.convert('RGB')))
                    images += 1
                else:
                    raise ValueError('unsupported content part; nothing is silently dropped')
            message['content'] = parts
        elif not isinstance(content, str):
            raise ValueError('message content must be text or content parts')
    if set(calls) != results:
        raise ValueError('history has tool calls without results')
    if images > 12:
        raise ValueError('more than 12 images; use harness context management')
    return messages, tools, images


def parse_reply(raw, *, truncated=False):
    """Parse only actual assistant recipient channels, never tool examples in thought.

    Generation starts immediately after the template's '<|start|>assistant'.
    Truncation rejects every call in that reply, including earlier complete ones.
    """
    message = dict(role='assistant', content='')
    if truncated:
        return message, 'length', ['truncated response; no calls forwarded']
    text = raw if raw.startswith('<|start|>') else '<|start|>assistant' + raw
    text = re.sub(r'<\|end_of_text\|>$', '', text)
    channels = list(re.finditer(
        r'<\|start\|>assistant(?P<header>[^<]*)<\|message\|>(?P<body>.*?)(?:<\|eom\|>|<\|eot\|>)',
        text, re.DOTALL))
    calls, thoughts, content, errors = [], [], [], []
    for channel in channels:
        recipient = re.search(r'\bto=([^\s]+)', channel['header'])
        recipient = recipient[1] if recipient else 'user'
        value = channel['body']
        if recipient == 'self':
            thoughts.append(value)
            continue
        if recipient == 'user':
            content.append(value)
            continue
        if recipient != 'python':
            errors.append(f'unsupported recipient: {recipient}')
            continue
        # Match whole invocation bodies. Do not use an XML parser or unescape code.
        invocations = list(re.finditer(r'<atem:invoke\s+name="([^"]+)">\s*(.*?)\s*</atem:invoke>', value, re.DOTALL))
        if not invocations or value.count('<atem:invoke') != len(invocations):
            errors.append('incomplete python invocation')
        for invoke in invocations:
            if invoke[1] != 'python':
                errors.append('invocation name does not match python recipient')
                continue
            parameter = re.fullmatch(r'<atem:parameter\s+name="input">(.*?)</atem:parameter>', invoke[2], re.DOTALL)
            if parameter is None:
                errors.append('python requires exactly one raw input parameter')
                continue
            calls.append(dict(id='call_' + uuid.uuid4().hex[:24], type='function',
                              function=dict(name='python', arguments=json.dumps({'input': parameter[1]}))))
    # Reject a dangling channel even if earlier channels contain complete calls.
    if not channels or text[channels[-1].end():].strip():
        errors.append('unparsed or incomplete assistant channel')
    if errors:
        # Let the harness score this as a model reply with no executable tool call.
        message['content'] = raw
        return message, 'stop', errors
    message['content'] = '\n'.join(content)
    if thoughts:
        message['reasoning_content'] = '\n'.join(thoughts)
    if calls:
        message['tool_calls'] = calls
    return message, 'tool_calls' if calls else 'stop', []


def completion(raw, *, truncated=False, prompt_tokens=0, completion_tokens=0, model='glimmer'):
    import time
    message, finish, errors = parse_reply(raw, truncated=truncated)
    return dict(id='chatcmpl-' + uuid.uuid4().hex, object='chat.completion', created=int(time.time()),
                model=model, choices=[dict(index=0, message=message, finish_reason=finish)],
                usage=dict(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                           total_tokens=prompt_tokens + completion_tokens),
                glimmer_bridge=dict(parse_errors=errors))
