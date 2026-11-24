"""One VST7 experiment: import generated pairs, train, evaluate and export candidates."""
import argparse
import hashlib
import json
from pathlib import Path

from .data import cache_native_dry, prepare, write_json
from .evaluate import evaluate, renderer_status
from .train import train

DEFAULT_OUTPUT = 'outputs/step2_ir_v1'


def check_cache_source(dataset, cache):
    """기존 캐시가 다른 데이터(예: 볼륨 보정 전)로 만들어졌으면 재사용하지 않고 중단."""
    metadata = json.loads((Path(cache)/'dataset.json').read_text(encoding='utf-8'))
    current = hashlib.sha256((Path(dataset)/'manifest.json').read_bytes()).hexdigest()
    if metadata.get('source_manifest_sha256') != current:
        raise ValueError(f'{cache} was built from a different dataset manifest; '
                         'use a new --output folder')


def run(dataset,output=DEFAULT_OUTPUT,epochs=30,batch_size=16,threads=4,
        evaluate_only=False,render_samples=70):
    output = Path(output)
    cache = output/'dataset'
    if (cache/'dataset.json').exists():
        check_cache_source(dataset, cache)
    else:
        if evaluate_only:
            raise FileNotFoundError('Prepare the dataset before evaluation')
        prepare(dataset,cache)
    if evaluate_only:
        checkpoint = output/'model/best.pt'
    else:
        cache_native_dry(cache)
        renderer_status(cache,refresh=True)
        checkpoint = train(cache,output/'model',epochs,batch_size,threads=threads)
    results = {split:evaluate(cache,checkpoint,output/split,split,batch_size,render_samples)
               for split in ('validation','test')}
    write_json(output/'summary.json',results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--output',default=DEFAULT_OUTPUT)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--evaluate-only',action='store_true')
    parser.add_argument('--render-samples',type=int,default=70)
    print(json.dumps(run(**vars(parser.parse_args())),indent=2))


if __name__=='__main__':
    main()
