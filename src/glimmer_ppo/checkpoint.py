"""Update-boundary checkpoints; interrupted episodes restart on resume."""
import json
from pathlib import Path
import random
import re
import uuid

import torch


def checkpoint_config(path):
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f'Checkpoint directory does not exist: {path}')
    for config in (path / 'training_config.json', path.parent.parent / 'config.json'):
        if config.is_file():
            return json.loads(config.read_text())
    raise ValueError(f'No training configuration found for {path}')


def checkpoint_update(path):
    path = Path(path)
    meta = path / 'checkpoint.json'
    if meta.exists():
        return int(json.loads(meta.read_text())['update'])
    match = re.fullmatch(r'update_(\d+)', path.name)
    if not match:
        raise ValueError('Legacy checkpoint directory must be named update_XXXX')
    return int(match.group(1))


def save_checkpoint(policy, optimizer, sampler, path, update, config):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Checkpoint already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name('.' + path.name + '-' + uuid.uuid4().hex + '.tmp')
    staging.mkdir()
    policy.save(staging)
    state = dict(update=update, optimizer=optimizer.state_dict(),
                 python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
                 sampler_rng=sampler.rng.getstate() if hasattr(sampler, 'rng') else None)
    torch.save(state, staging / 'training_state.pt')
    (staging / 'training_config.json').write_text(json.dumps(config, indent=2))
    (staging / 'checkpoint.json').write_text(json.dumps(dict(update=update, format_version=1)))
    staging.rename(path)


def restore_checkpoint(policy, optimizer, sampler, path, log=print):
    path = Path(path)
    update = checkpoint_update(path)
    policy.load(path)
    state_path = path / 'training_state.pt'
    if not state_path.exists():
        log('[ppo] legacy checkpoint: restored LoRA and value head; optimizer and RNG states '
            'were not saved and start fresh')
        return update
    state = torch.load(state_path, map_location='cpu', weights_only=True)
    if state['update'] != update:
        raise ValueError('Checkpoint update metadata does not match training state')
    optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['python_rng'])
    torch.set_rng_state(state['torch_rng'])
    if state['sampler_rng'] is not None and hasattr(sampler, 'rng'):
        sampler.rng.setstate(state['sampler_rng'])
    if state['cuda_rng']:
        if torch.cuda.is_available() and len(state['cuda_rng']) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        else:
            log('[ppo] GPU count changed; CUDA RNG state starts fresh')
    log('[ppo] restored optimizer, update counter, and saved RNG states')
    return update
