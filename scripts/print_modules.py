"""Print Muse's linear layers and which ones LoRA will adapt. Loads on the meta device (no weights).

    python scripts/print_modules.py
"""
import argparse
from collections import Counter

import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoConfig
import transformers

from glimmer_ppo.policy import FROZEN_PARTS, PolicyConfig, lora_target_names


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=PolicyConfig.model)
    args = p.parse_args()
    config = AutoConfig.from_pretrained(args.model)
    cls = (getattr(transformers, 'MuseGlimmerForConditionalGeneration', None)
           or getattr(transformers, 'AutoModelForImageTextToText'))
    with init_empty_weights():
        model = cls._from_config(config) if hasattr(cls, '_from_config') else cls.from_config(config)
    leaves = Counter()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            frozen = any(part in name.lower() for part in FROZEN_PARTS)
            leaves[(name.rsplit('.', 1)[-1], 'frozen part' if frozen else 'text model')] += 1
    for (leaf, where), n in sorted(leaves.items()):
        print(f'{leaf:20s} {where:12s} x{n}')
    targets = lora_target_names(model, set(PolicyConfig.lora_targets))
    print(f'\nLoRA targets: {len(targets)} layers, e.g. {targets[:3]}')


if __name__ == '__main__':
    main()
