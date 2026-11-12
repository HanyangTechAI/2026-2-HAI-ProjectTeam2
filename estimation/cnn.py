"""Shared-encoder CNN baseline; not a reproduction of a specific paper."""
import torch
from torch import nn
from torch.nn import functional as F

from .chain import continuous, discrete


class PairCNN(nn.Module):
    def __init__(self, definitions, channels=(16, 32, 64, 128), hidden=256,
                 task='joint', effect_types=None, include_bypass=True):
        super().__init__()
        if task not in ('joint', 'classification', 'regression'):
            raise ValueError('Unknown CNN task')
        self.task = task
        self.definitions = definitions
        self.effect_types = list(definitions) if effect_types is None else list(effect_types)
        if not self.effect_types or len(set(self.effect_types)) != len(self.effect_types) or set(self.effect_types) - set(definitions):
            raise ValueError('Expected unique supported effect types')
        if task == 'regression' and (include_bypass or len(self.effect_types) != 1):
            raise ValueError('Regression requires one known effect without bypass')
        self.classes = (['bypass'] if include_bypass else []) + self.effect_types
        self.model_config = {'channels': list(channels), 'hidden': hidden, 'task': task,
                             'effect_types': self.effect_types, 'include_bypass': include_bypass}
        blocks, previous = [], 1
        for width in channels:
            blocks.extend([nn.Conv2d(previous, width, 3, padding=1),
                           nn.GroupNorm(4, width), nn.ReLU(), nn.MaxPool2d(2)])
            previous = width
        self.encoder = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fusion = nn.Sequential(nn.Linear(previous * 4, hidden), nn.ReLU())
        self.classifier = nn.Linear(hidden, len(self.classes)) if task != 'regression' else None
        self.knobs = nn.ModuleDict({t: nn.Linear(hidden, len(continuous(d)))
                                    for t, d in definitions.items()
                                    if t in self.effect_types and continuous(d) and task != 'classification'})
        self.discrete_heads = nn.ModuleDict({
            f'{t}__{p["name"]}': nn.Linear(hidden, 1 if p['type'] == 'boolean' else len(p['options']))
            for t, d in definitions.items() for p in discrete(d)
            if t in self.effect_types and task != 'classification'})

    def forward(self, dry, reference):
        zd, zr = self.encoder(dry), self.encoder(reference)
        delta = zr - zd
        h = self.fusion(torch.cat([zd, zr, delta, delta.abs()], dim=1))
        return {'task': self.task, 'logits': self.classifier(h) if self.classifier is not None else None,
                'continuous': {t: head(h).sigmoid() for t, head in self.knobs.items()},
                'discrete': {key: head(h) for key, head in self.discrete_heads.items()}}


def masked_loss(output, labels, knobs, discrete_targets, definitions, classes,
                class_weight=1.0, knob_weight=1.0, discrete_weight=1.0):
    task = output.get('task', 'joint')
    anchor = output['logits'] if output['logits'] is not None else next(iter(output['continuous'].values()))
    zero = anchor.sum() * 0
    classification = F.cross_entropy(output['logits'], labels) if task != 'regression' else zero
    # Differentiable zero handles a batch consisting entirely of bypass examples.
    knob_sum = zero
    discrete_sum = zero
    knob_count = discrete_count = 0
    for t, definition in definitions.items():
        if t not in classes or task == 'classification':
            continue
        mask = labels == classes.index(t)
        count = int(mask.sum())
        if not count:
            continue
        if continuous(definition):
            pred, target = output['continuous'][t][mask], knobs[t][mask]
            knob_sum = knob_sum + F.huber_loss(pred, target, reduction='sum')
            knob_count += target.numel()
        for p in discrete(definition):
            key = f'{t}__{p["name"]}'
            pred, target = output['discrete'][key][mask], discrete_targets[key][mask]
            if p['type'] == 'boolean':
                loss = F.binary_cross_entropy_with_logits(pred[:, 0], target.float(), reduction='sum')
            else:
                loss = F.cross_entropy(pred, target, reduction='sum')
            discrete_sum = discrete_sum + loss
            discrete_count += count
    knob_loss = knob_sum / max(1, knob_count)
    discrete_loss = discrete_sum / max(1, discrete_count)
    total = class_weight * classification + knob_weight * knob_loss + discrete_weight * discrete_loss
    return total, {'classification': float(classification.detach()),
                   'knob_huber': float(knob_loss.detach()), 'discrete': float(discrete_loss.detach())}
