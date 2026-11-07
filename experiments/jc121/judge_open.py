"""Run released open judges on the blinded MIME answer pairs, using CPU inference."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

from judge_jev import CRITERIA, INSTRUCTIONS


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=['verdict', 'laya', 'kev', 'pplx'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=840)
    args = parser.parse_args()
    root, name = args.root, args.model
    config = json.loads((root / 'models.json').read_text())[name]
    pairs = json.loads((root / 'blind-pairs.json').read_text())[:args.limit]
    assert len({p['id'] for p in pairs}) == len(pairs)
    destination = root / name
    destination.mkdir(exist_ok=True)

    import torch
    from huggingface_hub import snapshot_download

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    checkpoint = snapshot_download(config['repo'], revision=config['revision'],
                                   ignore_patterns=['*.onnx', 'multilingual/*', 'typed-decisions/*'],
                                   max_workers=4)
    save(destination / 'environment.json', {
        'model': config, 'checkpoint': checkpoint, 'device': 'cpu',
        'threads': torch.get_num_threads(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
    })
    started = time.perf_counter()
    if name == 'verdict':
        sys.path.insert(0, str(root / 'sources/verdict'))
        from core.engine_encoder import DecisionEngine
        from core.primitives import Choice, Option
        from core.formatting import build_model_input, format_query
        engine = DecisionEngine(checkpoint, device='cpu', max_length=1024)
        question = Choice(id='correct', question=INSTRUCTIONS,
                          options=[Option(id=k, description=v) for k, v in CRITERIA.items()])

        def predict(request):
            state = json.dumps(request['state'], ensure_ascii=False)
            _, labels, _ = format_query(state, question)
            text = build_model_input(question.question, state, labels)
            token_count = len(engine.tokenizer(text)['input_ids'])
            assert token_count <= engine.max_length, 'Verdict input would truncate'
            response = engine.evaluate(state, [question]).model_dump()
            response['input_tokens'] = token_count
            return response, response['results'][0]['probabilities']['true']

    elif name == 'laya':
        import laya
        engine = laya.load(checkpoint, device='cpu', backend='eager', vegaml=False)

        def predict(request):
            response = engine.system_one(request['state'], request['questions'],
                                         max_len=1024, head_max_len=768)
            assert not response['usage'].get('truncated'), 'Laya state would truncate'
            assert not response['usage'].get('options'), 'Laya options collapsed'
            return response, response['answers']['correct']['noul']

    elif name == 'kev':
        sys.path.insert(0, str(root / 'sources/kev'))
        from kev.predictors import LocalPredictor
        from kev.checkpoint import LoadOptions
        engine = LocalPredictor(checkpoint, 'cpu', opts=LoadOptions(dtype=torch.float32))

        def predict(request):
            # The upstream evaluator requires a placeholder target, which its forward pass never reads.
            native = {**request, 'questions': {'correct': {**request['questions']['correct'], 'label': False, 'src': 'mime'}}}
            response = engine(native)
            return response, response['probabilities']['correct']['true']

    else:
        sys.path.insert(0, str(Path(checkpoint) / 'source/src'))
        from pplx_decider import Decider
        engine = Decider.from_pretrained(checkpoint, device='cpu', cpu_threads=8, batch_tokens=2048)

        def predict(request):
            response = engine.decide(request)
            assert response['usage']['sequences'] == 1
            return response, response['answers']['correct']['noul']

    load_seconds = time.perf_counter() - started
    print(name, 'loaded', round(load_seconds, 2), 'seconds', flush=True)
    execution_started = time.perf_counter()
    requests = []
    for i, pair in enumerate(pairs):
        request = {'state': {'reference_actions': pair['references'], 'candidate_answer': pair['answer']},
                   'questions': {'correct': {'type': 'noul', 'instructions': INSTRUCTIONS, 'criteria': CRITERIA}}}
        requests.append(request)
        target = destination / 'responses' / (pair['id'] + '.json')
        request_hash = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        if target.exists():
            old = json.loads(target.read_text())
            assert old['request_sha256'] == request_hash and old['model_revision'] == config['revision']
            continue
        before = time.perf_counter()
        with torch.inference_mode():
            response, probability = predict(request)
        assert isinstance(probability, (int, float)) and math.isfinite(probability) and 0 <= probability <= 1
        save(target, {'pair_id': pair['id'], 'request': request, 'request_sha256': request_hash,
                      'model_revision': config['revision'], 'probability': probability,
                      'credit': probability >= 0.5, 'response': response,
                      'elapsed_seconds': time.perf_counter() - before})
        if i % 20 == 0 or i + 1 == len(pairs):
            save(destination / 'progress.json', {'completed': i + 1, 'expected': len(pairs),
                 'elapsed_seconds': time.perf_counter() - execution_started})
            print(name, i + 1, '/', len(pairs), 'seconds', round(time.perf_counter() - execution_started, 1), flush=True)
    if args.limit == 840:
        timings, differences = [], []
        for repetition in range(5):
            before = time.perf_counter()
            for i in range(0, 840, 84):
                with torch.inference_mode():
                    _, probability = predict(requests[i])
                original = json.loads((destination / 'responses' / (pairs[i]['id'] + '.json')).read_text())
                differences.append(abs(probability - original['probability']))
            timings.append((time.perf_counter() - before) / 10)
        save(destination / 'timing.json', {'repetitions': 5, 'pairs_per_repetition': 10,
             'seconds_per_pair': timings, 'max_repeat_probability_difference': max(differences),
             'load_seconds_excluding_download': load_seconds, 'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    save(destination / 'complete.json', {'pairs': len(pairs), 'model': config,
         'elapsed_seconds': time.perf_counter() - execution_started,
         'load_seconds_excluding_download': load_seconds, 'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


if __name__ == '__main__':
    main()
