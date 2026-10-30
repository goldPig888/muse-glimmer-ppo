"""Read one robot tool call out of Muse's raw generated text.

With llama.cpp the server parsed tool calls; with Hugging Face generation we
get raw text, so this accepts the common layouts and takes the LAST call in
the reply (reasoning may mention earlier drafts):

* <tool_call>{"name": "move_to", "arguments": {...}}</tool_call>
* {"name": "move_to", "arguments": {...}}  (arguments may be a JSON string)
* {"type": "function", "function": {"name": ..., "arguments": ...}}
* move_to({...})

Run scripts/show_reply.py once on the real model and add Muse's exact
format here if it is different. A reply with no recognisable call is
returned as None and counts as one invalid call, as in the paper's runs.
"""
from __future__ import annotations

import json
import re

ROBOT_TOOLS = ('move_to', 'done', 'give_up')
_decoder = json.JSONDecoder()


def _as_call(obj):
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get('function'), dict):
        obj = obj['function']
    name = obj.get('name')
    if name not in ROBOT_TOOLS:
        return None
    args = obj.get('arguments', obj.get('parameters', {}))
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return None
    return (name, args) if isinstance(args, dict) else None


def _json_objects(text):
    i = 0
    while True:
        i = text.find('{', i)
        if i < 0:
            return
        try:
            obj, end = _decoder.raw_decode(text, i)
            yield obj, i, end
            i = end
        except json.JSONDecodeError:
            i += 1


def parse_tool_call(text: str):
    """Return (name, arguments) of the last robot tool call in ``text``, or None."""
    found = []
    for obj, start, _ in _json_objects(text):
        call = _as_call(obj)
        if call:
            found.append((start, call))
    for m in re.finditer(r'\b(move_to|done|give_up)\s*\(\s*', text):
        rest = text[m.end():]
        if rest.startswith(')'):
            found.append((m.start(), (m.group(1), {})))
            continue
        try:
            obj, _ = _decoder.raw_decode(rest)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            found.append((m.start(), (m.group(1), obj)))
    if not found:
        return None
    return max(found, key=lambda f: f[0])[1]
