"""Train the paired CNN from prepared dry/wet segments."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .audio import AudioConfig, log_mel
from .cnn import PairCNN, masked_loss
from .chain import catalog, continuous, discrete, fingerprint, normalize
from .data import load_manifest, pair_audio, write_json


class PairDataset(Dataset):
    def __init__(self, rows, metadata, split, cache_features=False):
        self.rows = [r for r in rows if r['split'] == split]
        if not self.rows:
            raise ValueError(f'Empty split: {split}')
        self.config = AudioConfig(**metadata['audio_config'])
        self.definitions = {t: metadata['catalog'][t] for t in metadata.get('effect_types', metadata['catalog'])}
        self.classes = metadata.get('classes', ['bypass', *self.definitions])
        self.cache_features = cache_features
        self.feature_cache = {}

    def features(self, row):
        if self.cache_features and row['dry_path'] in self.feature_cache and row['wet_path'] in self.feature_cache:
            return self.feature_cache[row['dry_path']], self.feature_cache[row['wet_path']]
        dry, wet = pair_audio(row, self.config)
        result = []
        for path, audio in ((row['dry_path'], dry), (row['wet_path'], wet)):
            feature = self.feature_cache.get(path) if self.cache_features else None
            if feature is None:
                feature = torch.from_numpy(log_mel(audio, self.config))
                if self.cache_features:
                    self.feature_cache[path] = feature
            result.append(feature)
        return tuple(result)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        dry, wet = self.features(row)
        items = row['ground_truth_chain']['chain']
        name = items[0]['effect_type'] if items else 'bypass'
        values = items[0]['values'] if items else {}
        knobs, targets = {}, {}
        for t, definition in self.definitions.items():
            knobs[t] = torch.tensor([normalize(values[p['name']], p) if name == t else 0
                                     for p in continuous(definition)], dtype=torch.float32)
            for p in discrete(definition):
                options = [False, True] if p['type'] == 'boolean' else p['options']
                targets[f'{t}__{p["name"]}'] = options.index(values[p['name']]) if name == t else 0
        return dry, wet, self.classes.index(name), knobs, targets


def train(manifest, output, epochs=20, batch_size=16, learning_rate=1e-3,
          seed=42, device='cpu', class_weight=1.0, knob_weight=1.0,
          discrete_weight=1.0, num_workers=0, threads=4, cache_features=False):
    if (epochs < 1 or batch_size < 1 or learning_rate <= 0 or threads < 1 or num_workers < 0
            or any(not np.isfinite(w) or w < 0 for w in (class_weight, knob_weight, discrete_weight))):
        raise ValueError('Invalid training configuration')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    rows, metadata = load_manifest(manifest)
    if fingerprint(catalog()) != metadata['catalog_hash']:
        raise ValueError('Renderer catalog changed; regenerate data')
    task = metadata.get('task', 'joint')
    model = PairCNN(metadata['catalog'], task=task, effect_types=metadata.get('effect_types'),
                    include_bypass=metadata.get('include_bypass', True)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    datasets = {s: PairDataset(rows, metadata, s, cache_features) for s in ('train', 'validation')}
    loaders = {s: DataLoader(d, batch_size=batch_size, shuffle=s == 'train', num_workers=num_workers,
                             generator=torch.Generator().manual_seed(seed)) for s, d in datasets.items()}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history, best = [], float('inf')
    training_config = dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                           seed=seed, device=device, class_weight=class_weight, knob_weight=knob_weight,
                           discrete_weight=discrete_weight, num_workers=num_workers, threads=threads,
                           cache_features=cache_features, task=task)
    for epoch in range(epochs):
        record = {'epoch': epoch + 1}
        for split, loader in loaders.items():
            model.train(split == 'train')
            total = correct = count = 0
            components = {'classification': 0.0, 'knob_huber': 0.0, 'discrete': 0.0}
            knob_error, knob_count = 0.0, 0
            with torch.set_grad_enabled(split == 'train'):
                for dry, wet, labels, knobs, targets in loader:
                    dry, wet, labels = dry.to(device), wet.to(device), labels.to(device)
                    knobs = {k: v.to(device) for k, v in knobs.items()}
                    targets = {k: v.to(device) for k, v in targets.items()}
                    prediction = model(dry, wet)
                    loss, stats = masked_loss(prediction, labels, knobs, targets, model.definitions, model.classes,
                                          class_weight, knob_weight, discrete_weight)
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite training loss')
                    if split == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    total += float(loss.detach()) * len(labels)
                    for key, value in stats.items():
                        components[key] += value * len(labels)
                    if prediction['logits'] is not None:
                        correct += int((prediction['logits'].argmax(1) == labels).sum())
                    for t, values in prediction['continuous'].items():
                        mask = labels == model.classes.index(t)
                        active = values[mask].detach()
                        knob_error += float((active - knobs[t][mask]).abs().sum())
                        knob_count += active.numel()
                    count += len(labels)
            record[split] = {'loss': total / count, 'accuracy': correct / count if task != 'regression' else None,
                             'classification_loss': components['classification'] / count if task != 'regression' else None,
                             'knob_huber': components['knob_huber'] / count if task != 'classification' else None,
                             'normalized_mae': knob_error / knob_count if knob_count else None}
        history.append(record)
        print(json.dumps(record), flush=True)
        checkpoint = {'format_version': 1, 'state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch + 1,
                      'model_config': model.model_config, 'classes': model.classes,
                      'catalog': model.definitions, 'catalog_hash': metadata['catalog_hash'],
                      'audio_config': metadata['audio_config'], 'renderer': metadata['renderer'],
                      'data_versions': metadata['versions'],
                      'parameter_order': {t: [p['name'] for p in continuous(model.definitions[t])]
                                          for t in model.effect_types if task != 'classification'},
                      'normalization': 'schema linear or natural-log range -> [0,1]; inverse identical in CMA',
                      'training_config': training_config, 'validation': record['validation'],
                      'manifest_sha256': hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
                      'torch_version': str(torch.__version__)}
        torch.save(checkpoint, output / 'last.pt')
        if record['validation']['loss'] < best:
            best = record['validation']['loss']
            torch.save(checkpoint, output / 'best.pt')
        write_json(output / 'history.json', history)
    return output / 'best.pt'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--learning-rate', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cpu')
    p.add_argument('--class-weight', type=float, default=1)
    p.add_argument('--knob-weight', type=float, default=1)
    p.add_argument('--discrete-weight', type=float, default=1)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--cache-features', action='store_true', help='Cache log-mel features in RAM during this run')
    print(train(**vars(p.parse_args())))


if __name__ == '__main__':
    main()
