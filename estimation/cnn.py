"""Paired CNN: seven VST classes, optional stages, knobs and distance auxiliary task."""
import torch
from torch import nn
from torch.nn import functional as F

from .chain import AUXILIARY, DRIVES, continuous


class PairCNN(nn.Module):
    def __init__(self, definitions, channels=(8, 16, 32, 64), hidden=256):
        super().__init__()
        self.definitions = definitions
        self.classes = list(DRIVES)
        self.model_config = {'channels': list(channels), 'hidden': hidden}
        blocks, previous = [], 1
        for width in channels:
            blocks += [nn.Conv2d(previous, width, 3, padding=1), nn.GroupNorm(4, width),
                       nn.ReLU(), nn.MaxPool2d(2)]
            previous = width
        # Keep four frequency regions instead of averaging all frequency cues away.
        self.encoder = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d((4, 1)), nn.Flatten())
        self.fusion = nn.Sequential(nn.Linear(previous * 16, hidden), nn.ReLU())
        self.classifier = nn.Linear(hidden, len(DRIVES))
        self.presence = nn.Linear(hidden, len(AUXILIARY))
        self.knobs = nn.ModuleDict({t: nn.Linear(hidden, len(continuous(d)))
                                    for t, d in definitions.items()})
        self.distance = nn.Linear(hidden, 2)

    def forward(self, dry, reference):
        zd, zr = self.encoder(torch.cat((dry,reference),dim=0)).chunk(2,dim=0)
        delta = zr - zd
        h = self.fusion(torch.cat((zd, zr, delta, delta.abs()), dim=1))
        return {'logits': self.classifier(h), 'presence': self.presence(h),
                'knobs': {t: layer(h).sigmoid() for t, layer in self.knobs.items()},
                'distance': F.softplus(self.distance(h))}


def supervised_loss(output, batch, distance_weight=0.1):
    classification = F.cross_entropy(output['logits'], batch['label'])
    presence = F.binary_cross_entropy_with_logits(output['presence'], batch['presence'])
    total, count = classification * 0, 0
    for t, prediction in output['knobs'].items():
        mask = batch['active'][t].bool()
        if mask.any():
            total = total + F.huber_loss(prediction[mask], batch['knobs'][t][mask], reduction='sum')
            count += prediction[mask].numel()
    knobs = total / max(1, count)
    # This is d(generated dry, target wet), NOT d(rendered predicted knobs, target).
    distance = F.huber_loss(output['distance'], batch['distance'])
    loss = classification + presence + knobs + distance_weight * distance
    return loss, {'classification': classification, 'presence': presence,
                  'knobs': knobs, 'distance_auxiliary': distance}
