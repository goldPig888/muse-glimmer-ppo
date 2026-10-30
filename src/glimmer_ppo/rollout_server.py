"""Generate PPO rollouts with llama.cpp on one GPU while Hugging Face trains on the other.

Why: sampling replies is most of PPO's time, and llama.cpp generates much
faster than ``transformers.generate``. Layout on 2x A6000:

    GPU 0: llama-server, Muse GGUF (Q8_0 recommended) + mmproj + current LoRA
    GPU 1: Hugging Face QLoRA training (scoring, value head, updates)

How it stays correct:
* The prompt is rendered by the Hugging Face chat template (same tools, same
  reasoning setting) and sent raw to ``/completion``; image placeholders are
  swapped for llama.cpp's media marker.
* llama.cpp returns the sampled token ids; PPO scores exactly those ids with
  the training model (``Policy.record_for`` + ``Policy.score``). Both use
  Muse's 202,048-token vocabulary.
* Sampling uses the full distribution (top_k=0, top_p=1, min_p=0) at the
  training temperature, so it matches the policy PPO assumes.
* After each PPO update the LoRA is converted to GGUF and the server is
  restarted with it, so rollouts always come from the latest policy.

What is approximate: the server runs a GGUF quantization (Q8_0 or Q4) while
training runs NF4 4-bit, so samples are very slightly off-policy. PPO's ratio
is still computed consistently by the training model. Q8_0 keeps the gap small.
"""
from __future__ import annotations

import base64
import io
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

MEDIA_MARKER = '<__media__>'   # llama.cpp mtmd default marker


@dataclass
class ServerConfig:
    server_bin: str                 # path to llama-server
    gguf: str                       # Muse GGUF (Q8_0 recommended)
    mmproj: str                     # vision projector GGUF
    gpu: str = '0'                  # physical GPU for the server
    port: int = 8090
    ctx: int = 32768
    extra_args: tuple = ()
    convert_script: str | None = None   # llama.cpp convert_lora_to_gguf.py
    hf_base_dir: str | None = None      # local HF snapshot of the base model (for LoRA conversion)


def png_b64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def to_server_prompt(text: str, image_token: str, n_images: int) -> str:
    """Swap the HF template's image placeholders for llama.cpp's marker."""
    found = text.count(image_token)
    if found != n_images:
        raise ValueError(f'prompt has {found} image placeholders ({image_token!r}) but {n_images} images; '
                         'check Policy.image_token against the chat template')
    return text.replace(image_token, MEDIA_MARKER)


def parse_completion(payload: dict, max_tokens: int):
    """-> (token ids, text, truncated) from a /completion response."""
    tokens = payload.get('tokens')
    if not tokens:
        raise RuntimeError('llama-server returned no token ids; this llama.cpp is too old for return_tokens')
    stop_type = payload.get('stop_type')
    truncated = stop_type == 'limit' if stop_type is not None else bool(payload.get('stopped_limit'))
    truncated = truncated or (len(tokens) >= max_tokens and not payload.get('stopped_eos', False))
    return [int(t) for t in tokens], payload.get('content', ''), truncated


class LlamaServer:
    def __init__(self, cfg: ServerConfig, log_dir: Path):
        self.cfg = cfg
        self.log_dir = Path(log_dir)
        self.proc: subprocess.Popen | None = None
        self.adapter: str | None = None
        self.url = f'http://127.0.0.1:{cfg.port}'
        self.client = httpx.Client(trust_env=False, timeout=1800)

    # -- lifecycle -------------------------------------------------------
    def start(self, adapter_gguf: str | None = None):
        self.stop()
        self.adapter = adapter_gguf
        cmd = [self.cfg.server_bin, '-m', self.cfg.gguf, '--mmproj', self.cfg.mmproj,
               '--port', str(self.cfg.port), '--host', '127.0.0.1', '-c', str(self.cfg.ctx),
               '-ngl', '999', '--jinja', *self.cfg.extra_args]
        if adapter_gguf:
            cmd += ['--lora', adapter_gguf]
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': self.cfg.gpu}
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log = (self.log_dir / 'llama_server.log').open('a')
        log.write(f'\n=== {time.ctime()} {" ".join(cmd)}\n')
        log.flush()
        self.proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.time() + 900
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f'llama-server exited; see {self.log_dir / "llama_server.log"}')
            try:
                if self.client.get(self.url + '/health', timeout=5).status_code == 200:
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(2)
        raise RuntimeError('llama-server did not become healthy in 15 minutes')

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    # -- generation --------------------------------------------------------
    def generate(self, prompt: str, images, max_tokens: int, temperature: float, seed: int | None = None):
        body = {
            'prompt': {'prompt_string': prompt, 'multimodal_data': [png_b64(im) for im in images]},
            'n_predict': max_tokens, 'temperature': temperature,
            'top_k': 0, 'top_p': 1.0, 'min_p': 0.0, 'typical_p': 1.0,
            'repeat_penalty': 1.0, 'presence_penalty': 0.0, 'frequency_penalty': 0.0,
            'return_tokens': True, 'cache_prompt': True, 'special': True,
        }
        if seed is not None:
            body['seed'] = seed
        r = self.client.post(self.url + '/completion', json=body)
        r.raise_for_status()
        return parse_completion(r.json(), max_tokens)

    # -- adapter sync --------------------------------------------------------
    def load_adapter(self, peft_dir: Path) -> str:
        """Convert a saved PEFT adapter to GGUF and restart the server with it."""
        if not (self.cfg.convert_script and self.cfg.hf_base_dir):
            raise RuntimeError('set convert_script and hf_base_dir to sync the LoRA into llama.cpp')
        out = Path(peft_dir) / 'adapter.gguf'
        cmd = [sys.executable, self.cfg.convert_script, '--base', self.cfg.hf_base_dir,
               '--outfile', str(out), '--outtype', 'f16', str(peft_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError('LoRA conversion failed:\n' + result.stderr[-4000:])
        self.start(str(out))
        return str(out)


def generate_with_server(server: LlamaServer, policy, messages, tools):
    """Sample one reply on llama.cpp and build the PPO record from its token ids.

    Returns (text, truncated, record), the same shape as ``Policy.act``."""
    prompt, images = policy.render_prompt(messages, tools)
    prompt = to_server_prompt(prompt, policy.image_token, len(images))
    bos = policy.processor.tokenizer.bos_token
    if bos and prompt.startswith(bos):
        prompt = prompt[len(bos):]          # llama.cpp adds BOS itself
    ids, _, truncated = server.generate(prompt, images, policy.cfg.max_new_tokens, policy.cfg.temperature)
    text = policy.processor.tokenizer.decode(ids, skip_special_tokens=False)
    return text, truncated, policy.record_for(messages, tools, ids)
