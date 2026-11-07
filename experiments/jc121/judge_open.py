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
    parser.add_argument('--tune-threads', action='store_true')
    parser.add_argument('--batch-size', type=int, default=1)
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
        engine = Decider.from_pretrained(checkpoint, device='cpu', cpu_threads=8, batch_tokens=8192)

        def predict(request):
            response = engine.decide(request)
            assert response['usage']['sequences'] == 1
            return response, response['answers']['correct']['noul']

    def predict_batch(requests):
        if name == 'kev':
            from kev.api import SystemOneRequest, to_record
            encodings = [engine.model.encode(engine.tok, to_record(SystemOneRequest(**r))[0],
                         max_state=engine.context['max_state'], max_branch=engine.context['max_branch'], strict=True)
                         for r in requests]
            assert all(len(e['ids']) <= engine.context['max_packed'] for e in encodings)
            logits = engine.model.forward_batch(encodings)
            results = []
            for encoded, result in zip(encodings, logits, strict=True):
                z = result[0].float().cpu()
                probability = torch.softmax(z, -1).tolist()
                response = {'probabilities': {'correct': dict(zip(['false', 'true'], probability))},
                            'logits': {'correct': dict(zip(['false', 'true'], z.tolist()))},
                            'inference_temperature': engine.temperature, 'input_tokens': len(encoded['ids'])}
                results.append((response, probability[1]))
            return results
        if name == 'pplx':
            results = engine.decide(requests)
            assert all(r['usage']['sequences'] == 1 for r in results)
            return [(r, r['answers']['correct']['noul']) for r in results]
        return [predict(r) for r in requests]

    load_seconds = time.perf_counter() - started
    print(name, 'loaded', round(load_seconds, 2), 'seconds', flush=True)
    if args.tune_threads and (destination / 'thread-timing.json').exists():
        torch.set_num_threads(json.loads((destination / 'thread-timing.json').read_text())['selected_threads'])
    elif args.tune_threads:
        thread_timings = {8: [], 16: [], 32: []}
        trial_predictions = {8: [], 16: [], 32: []}
        pair = pairs[0]
        probe = {'state': {'reference_actions': pair['references'], 'candidate_answer': pair['answer']},
                 'questions': {'correct': {'type': 'noul', 'instructions': INSTRUCTIONS, 'criteria': CRITERIA}}}
        for repetition in range(6):
            for threads in thread_timings:
                torch.set_num_threads(threads)
                before = time.perf_counter()
                with torch.inference_mode():
                    _, probability = predict(probe)
                if repetition:
                    thread_timings[threads].append(time.perf_counter() - before)
                    trial_predictions[threads].append(probability)
        import statistics
        selected = min(thread_timings, key=lambda t: statistics.median(thread_timings[t]))
        assert len({p >= 0.5 for values in trial_predictions.values() for p in values}) == 1
        torch.set_num_threads(selected)
        save(destination / 'thread-timing.json', {'seconds': thread_timings, 'probabilities': trial_predictions,
                                                'selected_threads': selected, 'selection_metric': 'median seconds'})
        print(name, 'selected CPU threads', selected, flush=True)
    requests = [{'state': {'reference_actions': pair['references'], 'candidate_answer': pair['answer']},
                   'questions': {'correct': {'type': 'noul', 'instructions': INSTRUCTIONS, 'criteria': CRITERIA}}}
                for pair in pairs]
    if args.batch_size > 1:
        probe = requests[:4]
        with torch.inference_mode():
            singles = [predict(r)[1] for r in probe]
            batched = [p for _, p in predict_batch(probe)]
        assert all((a >= .5) == (b >= .5) and abs(a - b) < 1e-4 for a, b in zip(singles, batched, strict=True))
        save(destination / 'batch-validation.json', {'single_probabilities': singles,
             'batch_probabilities': batched, 'batch_size': args.batch_size, 'passed': True})
        print(name, 'batch validation passed', args.batch_size, flush=True)
    execution_started = time.perf_counter()
    for start in range(0, len(pairs), args.batch_size):
        pending = []
        for i in range(start, min(start + args.batch_size, len(pairs))):
            target = destination / 'responses' / (pairs[i]['id'] + '.json')
            request_hash = hashlib.sha256(json.dumps(requests[i], sort_keys=True).encode()).hexdigest()
            if target.exists():
                old = json.loads(target.read_text())
                assert old['request_sha256'] == request_hash and old['model_revision'] == config['revision']
            else:
                pending.append((i, target, request_hash))
        if not pending:
            continue
        before = time.perf_counter()
        with torch.inference_mode():
            results = predict_batch([requests[i] for i, _, _ in pending])
        batch_seconds = time.perf_counter() - before
        for (i, target, request_hash), (response, probability) in zip(pending, results, strict=True):
            assert isinstance(probability, (int, float)) and math.isfinite(probability) and 0 <= probability <= 1
            save(target, {'pair_id': pairs[i]['id'], 'request': requests[i], 'request_sha256': request_hash,
                          'model_revision': config['revision'], 'probability': probability,
                          'credit': probability >= 0.5, 'response': response,
                          'cpu_threads': torch.get_num_threads(), 'batch_size': len(pending),
                          'elapsed_seconds': batch_seconds / len(pending)})
        completed = min(start + args.batch_size, len(pairs))
        if completed % 40 < args.batch_size or completed == len(pairs):
            save(destination / 'progress.json', {'completed': completed, 'expected': len(pairs),
                 'elapsed_seconds': time.perf_counter() - execution_started})
            print(name, completed, '/', len(pairs), 'seconds', round(time.perf_counter() - execution_started, 1), flush=True)
    if args.limit == 840:
        timings, differences = [], []
        for repetition in range(5):
            before = time.perf_counter()
            indices = list(range(0, 840, 84))
            for start in range(0, len(indices), args.batch_size):
                batch = indices[start:start + args.batch_size]
                with torch.inference_mode():
                    predictions = predict_batch([requests[i] for i in batch])
                for i, (_, probability) in zip(batch, predictions, strict=True):
                    original = json.loads((destination / 'responses' / (pairs[i]['id'] + '.json')).read_text())
                    differences.append(abs(probability - original['probability']))
            timings.append((time.perf_counter() - before) / 10)
        save(destination / 'timing.json', {'repetitions': 5, 'pairs_per_repetition': 10,
             'seconds_per_pair': timings, 'max_repeat_probability_difference': max(differences),
             'cpu_threads': torch.get_num_threads(),
             'batch_size': args.batch_size,
             'load_seconds_excluding_download': load_seconds, 'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    save(destination / 'complete.json', {'pairs': len(pairs), 'model': config,
         'elapsed_seconds': time.perf_counter() - execution_started,
         'load_seconds_excluding_download': load_seconds, 'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


if __name__ == '__main__':
    main()
