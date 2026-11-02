#!/usr/bin/env python3
"""Serve frozen Glimmer for CaPtain's python tool. No Python code executes here."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

from glimmer_ppo.captain_bridge import completion, normalize_request


class FrozenGlimmer:
    def __init__(self, args):
        import torch
        from transformers import AutoProcessor
        from glimmer_ppo.policy import PolicyConfig, _load_base
        self.torch, self.args = torch, args
        torch.manual_seed(args.seed)
        self.cfg = PolicyConfig(model=args.model, load_in_4bit=args.load_in_4bit,
                                gradient_checkpointing=False, reasoning_strength=args.reasoning)
        self.processor = AutoProcessor.from_pretrained(args.model)
        self.model = _load_base(self.cfg).eval()
        self.model.requires_grad_(False)
        self.lock = threading.Lock()
        self.device = self.model.get_input_embeddings().weight.device
        print('[glimmer] frozen base model loaded; no adapters or critic', flush=True)
        print('[glimmer] device map: ' + str(getattr(self.model, 'hf_device_map', self.device)), flush=True)

    def generate(self, body):
        torch = self.torch
        if body.get('model') != self.args.served_name:
            raise ValueError('model does not match the served name')
        messages, tools, images = normalize_request(body)
        limit = body.get('max_tokens', self.args.max_new_tokens)
        if not isinstance(limit, int) or not 1 <= limit <= self.args.max_new_tokens:
            raise ValueError(f'max_tokens must be in [1, {self.args.max_new_tokens}]')
        temperature = float(body.get('temperature', 1.0))
        if not 0 < temperature <= 2 or body.get('top_p', 1.0) != 1.0:
            raise ValueError('requires 0 < temperature <= 2 and top_p=1 (full-distribution sampling)')
        if body.get('chat_template_kwargs', {}).get('enable_thinking') is False:
            raise ValueError('thinking-off has no verified Muse equivalent; use the configured reasoning strength')
        # Only the harness compacts history. Never silently truncate prompt/images here.
        with self.lock, torch.inference_mode():
            inputs = self.processor.apply_chat_template(
                messages, tools=tools, reasoning_strength=self.args.reasoning,
                add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt')
            n = inputs['input_ids'].shape[1]
            if n + limit > self.args.context_tokens:
                raise ValueError(f'context_length_exceeded: {n} prompt + {limit} output > {self.args.context_tokens}; no history removed')
            inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            print(f'[glimmer] generating: prompt={n} images={images} output_cap={limit}', flush=True)
            from tqdm import tqdm
            from transformers import StoppingCriteria, StoppingCriteriaList
            with tqdm(total=limit, desc='Glimmer tokens', unit='tok', mininterval=1, dynamic_ncols=True) as bar:
                class Progress(StoppingCriteria):
                    def __call__(self, input_ids, scores, **kwargs):
                        bar.update(input_ids.shape[1] - n - bar.n)
                        return False
                autocast = torch.autocast('cuda', dtype=torch.bfloat16) if self.device.type == 'cuda' else nullcontext()
                with autocast:
                    output = self.model.generate(**inputs, max_new_tokens=limit, do_sample=True,
                                                 temperature=temperature, top_p=1.0, top_k=0,
                                                 stopping_criteria=StoppingCriteriaList([Progress()]))
            tokens = output[0, n:].cpu()
            eos = self.model.generation_config.eos_token_id or self.processor.tokenizer.eos_token_id
            eos = set(eos if isinstance(eos, list) else [eos])
            truncated = len(tokens) >= limit and int(tokens[-1]) not in eos
            raw = self.processor.tokenizer.decode(tokens, skip_special_tokens=False)
            result = completion(raw, truncated=truncated, prompt_tokens=n,
                                completion_tokens=len(tokens), model=self.args.served_name)
            return result, raw


def make_server(backend, host, port, output):
    """Backend injection is also used by the CPU HTTP integration test."""
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, value):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/v1/models':
                self.send_json(200, dict(object='list', data=[dict(id=backend.args.served_name,
                               object='model', owned_by='local', root=backend.args.model)]))
            elif self.path == '/health':
                self.send_json(200, dict(ready=True, policy='Glimmer', frozen=True))
            else:
                self.send_json(404, dict(error=dict(message='not found')))

        def do_POST(self):
            if self.path != '/v1/chat/completions':
                self.send_json(404, dict(error=dict(message='not found')))
                return
            start = time.monotonic()
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 64 * 1024 * 1024:
                    raise ValueError('invalid request size (maximum 64 MiB)')
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError('request must be a JSON object')
                result, raw = backend.generate(body)
                row = dict(seconds=round(time.monotonic() - start, 2),
                           raw_reply=raw, response=result)
                # No prompts or hidden simulator state are written by this server.
                with (output / 'requests.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                calls = result['choices'][0]['message'].get('tool_calls', [])
                print(f'[glimmer] reply: python_calls={len(calls)} '
                      f'finish={result["choices"][0]["finish_reason"]} '
                      f'errors={result["glimmer_bridge"]["parse_errors"]} seconds={row["seconds"]}', flush=True)
                self.send_json(200, result)
            except (ValueError, KeyError, TypeError) as exc:
                code = 'context_length_exceeded' if str(exc).startswith('context_length_exceeded:') else 'invalid_request'
                self.send_json(400, dict(error=dict(message=str(exc), type='invalid_request_error', code=code)))
            except Exception as exc:
                print(f'[glimmer] generation failed: {type(exc).__name__}: {exc}', flush=True)
                self.send_json(500, dict(error=dict(message=str(exc), type='generation_error')))

    return ThreadingHTTPServer((host, port), Handler)


def main():
    from glimmer_ppo.policy import PolicyConfig
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default=PolicyConfig.model)
    p.add_argument('--served-name', default='glimmer')
    p.add_argument('--load-in-4bit', action='store_true')
    p.add_argument('--reasoning', choices=['low', 'medium', 'high'], default='high')
    p.add_argument('--context-tokens', type=int, default=65536)
    p.add_argument('--max-new-tokens', type=int, default=4096)
    p.add_argument('--port', type=int, default=8600)
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error('--output must be fresh')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'server_config.json').write_text(json.dumps(vars(args), default=str, indent=2) + '\n')
    backend = FrozenGlimmer(args)
    server = make_server(backend, '127.0.0.1', args.port, args.output)
    print(f'[glimmer] ready at http://127.0.0.1:{args.port}/v1; Python runs in CaPtain, not here', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
