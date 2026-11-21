"""Train the VST7 paired CNN; retain generated-dry provenance and distance scope."""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .chain import DRIVES
from .cnn import PairCNN, supervised_loss
from .data import PairDataset, write_json


def device_batch(batch, device):
    return {k: device_batch(v,device) if isinstance(v,dict) else v.to(device)
            for k,v in batch.items()}


def run_epoch(model, loader, optimizer=None, distance_weight=.1):
    training = optimizer is not None
    model.train(training)
    stats = {'loss':0., 'classification':0., 'knobs':0.,
             'distance_auxiliary':0., 'correct':0,
             'knob_abs':0., 'knob_count':0, 'd_abs':0., 'dry_target_d':0.}
    n = 0
    for batch in loader:
        batch = device_batch(batch,next(model.parameters()).device)
        with torch.set_grad_enabled(training):
            output = model(batch['dry'],batch['wet'])
            loss, components = supervised_loss(output,batch,distance_weight)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                optimizer.step()
        count = len(batch['label'])
        n += count
        stats['loss'] += float(loss.detach())*count
        for key,value in components.items():
            stats[key] += float(value.detach())*count
        stats['correct'] += int((output['logits'].argmax(1)==batch['label']).sum())
        for t, values in output['knobs'].items():
            mask = batch['active'][t].bool()
            stats['knob_abs'] += float((values[mask].detach()-batch['knobs'][t][mask]).abs().sum())
            stats['knob_count'] += int(values[mask].numel())
        # Equal normalized weights. Both estimates and targets are dry/reference distances.
        predicted_d, target_d = output['distance'].mean(1), batch['distance'].mean(1)
        stats['d_abs'] += float((predicted_d.detach()-target_d).abs().sum())
        stats['dry_target_d'] += float(target_d.sum())
    return {**{k:stats[k]/n for k in ('loss','classification','knobs','distance_auxiliary')},
            'accuracy':stats['correct']/n,
            'normalized_knob_mae':stats['knob_abs']/max(1,stats['knob_count']),
            'dry_target_d_estimate_mae':stats['d_abs']/n,
            'mean_measured_dry_target_d':stats['dry_target_d']/n}


def train(cache, output, epochs=30, batch_size=16, learning_rate=1e-3,
          seed=42, threads=4, distance_weight=.1, render_every=5):
    if epochs<1 or batch_size<1 or learning_rate<=0 or distance_weight<0:
        raise ValueError('Invalid training settings')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    datasets = {s:PairDataset(cache,s) for s in ('train','validation')}
    metadata = datasets['train'].metadata
    loaders = {s:DataLoader(d,batch_size=batch_size,shuffle=s=='train',num_workers=0,
                             generator=torch.Generator().manual_seed(seed)) for s,d in datasets.items()}
    model = PairCNN(metadata['catalog'])
    from .amp import context_from_record
    from .chain import fingerprint
    model.amp_context = context_from_record(metadata)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=learning_rate*.05)
    output = Path(output)
    if (output/'history.json').exists():
        raise FileExistsError('Training results already exist')
    output.mkdir(parents=True,exist_ok=True)
    history, best = [], float('inf')
    config = dict(epochs=epochs,batch_size=batch_size,learning_rate=learning_rate,seed=seed,
                  threads=threads,distance_weight=distance_weight,render_every=render_every)
    from .evaluate import render_validation_probe, renderer_status
    status = renderer_status(cache)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1,user_api='blas')
    write_json(output/'renderer_status.json',status)
    for epoch in range(1,epochs+1):
        start = time.perf_counter()
        record = {'epoch':epoch,'learning_rate':optimizer.param_groups[0]['lr']}
        for split in ('train','validation'):
            record[split] = run_epoch(model,loaders[split],optimizer if split=='train' else None,distance_weight)
        if status['available'] and (epoch==1 or epoch%render_every==0 or epoch==epochs):
            record['validation_rendered_d'] = render_validation_probe(model,cache,limit=14)
            record['validation_rendered_d']['evaluated'] = True
        else:
            record['validation_rendered_d'] = {'available':status['available'],'evaluated':False,
                'reason':'not a rendering epoch' if status['available'] else status['reason']}
        scheduler.step()
        record['seconds'] = time.perf_counter()-start
        history.append(record)
        checkpoint = {'format_version':3,'chain_policy':'single_vst_drive_only',
                      'amp_context':model.amp_context,'amp_context_hash':fingerprint(model.amp_context),
                      'data_quality_policy':metadata['data_quality_policy'],
                      'source_render_audit_sha256':metadata['source_render_audit_sha256'],
                      'state_dict':model.state_dict(),'epoch':epoch,
                      'model_config':model.model_config,'catalog':metadata['catalog'],
                      'catalog_hash':metadata['catalog_hash'],'classes':list(DRIVES),
                      'audio_config':metadata['audio_config'],'distance_settings':metadata['distance_settings'],
                      'source_manifest_sha256':metadata['source_manifest_sha256'],
                      'input_policy':metadata['input_policy'],'training_config':config,
                      'validation':record['validation'],'torch_version':str(torch.__version__)}
        torch.save(checkpoint,output/'last.pt')
        if record['validation']['loss']<best:
            best = record['validation']['loss']
            torch.save(checkpoint,output/'best.pt')
        write_json(output/'history.json',history)
        print(json.dumps(record),flush=True)
    return output/'best.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--distance-weight',type=float,default=.1)
    print(train(**vars(parser.parse_args())))


if __name__ == '__main__':
    main()
