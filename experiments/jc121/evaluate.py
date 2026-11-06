"""Evaluate fixed MIME video frames through OpenRouter and preserve raw evidence."""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import math
import os
import random
import time
import threading
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path('/mnt/efs/justin/0_experiments/jc121')
GATEWAY = 'https://zochi-gateway.justin-6b3.workers.dev'
SCOPE = 'jc121-mime-20261106'
MODELS = [
    'openai/gpt-6.1-sol',
    'anthropic/claude-opus-5.5',
    'deepseek/deepseek-v4.1-flash-1017',
    'qwen/qwen3.8-flash',
    'z-ai/glm-5.3-flash',
    'minimax/minimax-m3.1-flash',
]
VARIANTS = ['base+blank@0', 'adversarial+misaligned@0']
QUESTION_SOURCE = Path(__file__).resolve().parents[2] / 'results/mime-cropped/zero-shot/all/raw_mcq_openai_gpt-4o-mini.json'
FREEFORM = '''What action is the person doing in this image/video? Describe the action in a single short phrase (under 5 words). 

This action is being 'mimed' meaning backgrounds or objects that are relevant may not be present. Think about only the *action* taking place in the video, and give a response for what it looks like the character is "acting out" or doing "charades" of. Your answer shpould be a short phrase (under 5 words), with no punctuation or answer prefix such as 'Answer:'.'''


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(args):
    import av
    from PIL import Image

    info = json.loads((ROOT / 'dataset_info.json').read_text())
    revision = info['sha']
    base = f'https://huggingface.co/datasets/wise-east/mime-cropped/resolve/{revision}/test/'
    metadata = ROOT / 'metadata.csv'
    if not metadata.exists():
        metadata.write_bytes(fetch(base + 'metadata.csv'))
    rows = list(csv.DictReader(io.StringIO(metadata.read_text())))
    historical = json.loads(QUESTION_SOURCE.read_text())['predictions']
    questions = {Path(item['video_path']).name.replace('_Justin_', '_j_').replace('_Natali_', '_n_'): item
                 for item in historical}
    selected = [row for row in rows if row['config_name'] in VARIANTS]
    assert len(selected) == 172
    assert len({row['sample_id'] for row in selected}) == 172
    actions = sorted({row['action'] for row in selected})
    assert len(actions) == 86
    chosen = set(actions[:args.actions])
    cases = []
    for row in selected:
        if row['action'] not in chosen:
            continue
        directory = ROOT / 'cases' / row['sample_id']
        directory.mkdir(parents=True, exist_ok=True)
        manifest = directory / 'case.json'
        if manifest.exists():
            case = json.loads(manifest.read_text())
            question = questions[row['file_name']]
            options = list(question['options'])
            label = options[ord(question['correct_answer']) - 65]
            assert label in row['label'].split('|')
            random.Random('jc121/' + row['action']).shuffle(options)
            case.update({'options': options, 'correct_letter': 'ABCD'[options.index(label)],
                         'question_source_sha256': digest(QUESTION_SOURCE.read_bytes())})
            write_json(manifest, case)
            cases.append(case)
            continue
        video = directory / 'video.mp4'
        if not video.exists():
            temporary = video.with_suffix('.tmp')
            temporary.write_bytes(fetch(base + row['file_name']))
            temporary.replace(video)
        frames = []
        with av.open(str(video)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            frame_count = 0
            for index, frame in enumerate(container.decode(stream)):
                frame_count += 1
                if (index + 1) % 30:
                    continue
                image = frame.to_image()
                image.thumbnail((512, 512), Image.Resampling.BICUBIC)
                output = directory / f'frame-{index:05}.jpg'
                image.save(output, format='JPEG')
                frames.append({'index': index, 'path': str(output),
                               'sha256': digest(output.read_bytes()), 'size': list(image.size)})
        assert frames, f'No sampled frames: {row["sample_id"]}'
        question = questions[row['file_name']]
        options = list(question['options'])
        label = options[ord(question['correct_answer']) - 65]
        assert label in row['label'].split('|')
        assert len(options) == len(set(options)) == 4
        random.Random('jc121/' + row['action']).shuffle(options)
        case = {**row, 'dataset_revision': revision, 'video_sha256': digest(video.read_bytes()),
                'fps': fps, 'total_frames': frame_count, 'frames': frames, 'options': options,
                'correct_letter': 'ABCD'[options.index(label)],
                'question_source_sha256': digest(QUESTION_SOURCE.read_bytes())}
        write_json(manifest, case)
        cases.append(case)
        print('Prepared', row['sample_id'], row['action'], row['config_name'], len(frames), flush=True)
    for action in chosen:
        assert len([case for case in cases if case['action'] == action]) == 2
    write_json(ROOT / 'cases.json', cases)
    write_json(ROOT / 'preparation.json', {'dataset_revision': revision,
               'metadata_sha256': digest(metadata.read_bytes()), 'actions': len(chosen),
               'videos': len(cases), 'frame_counts': [len(case['frames']) for case in cases]})


def request_body(case, model, answer_format):
    if answer_format == 'mcq':
        options = '\n'.join(f'{letter}. {option}' for letter, option in zip('ABCD', case['options']))
        prompt = ('What action is the person doing in this image/video? Choose the most accurate '
                  f'description from the options below.\n\n{options}\n\n'
                  'Respond with just a single letter (A, B, C, or D).')
    else:
        prompt = FREEFORM
    content = [{'type': 'text', 'text': prompt}]
    for frame in case['frames']:
        data = Path(frame['path']).read_bytes()
        assert digest(data) == frame['sha256']
        content.append({'type': 'image_url', 'image_url': {
            'url': 'data:image/jpeg;base64,' + base64.b64encode(data).decode(), 'detail': 'low'}})
    return {'model': model, 'messages': [{'role': 'user', 'content': content}],
            'max_tokens': 8192, 'reasoning': {'effort': 'medium'},
            'provider': {'require_parameters': True}, 'stream': False}


def predict(args):
    key = os.environ.get('ZOCHI_KEY', '').removeprefix('Bearer ').strip()
    if not key:
        raise SystemExit('ZOCHI_KEY is missing')
    headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json',
               'User-Agent': 'zochi-internal/1.0', 'x-zochi-scope-id': SCOPE,
               'x-zochi-source': 'mime-jc121'}
    scope = {'scope_id': SCOPE, 'parent_scope_id': None, 'limit_usd': args.budget,
             'reporting_label': 'jc121 MIME evaluation'}
    request = urllib.request.Request(GATEWAY + '/control/scopes',
                                     data=json.dumps(scope).encode(), headers=headers)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                write_json(ROOT / 'gateway-scope.json', json.load(response))
            break
        except urllib.error.HTTPError as error:
            if error.code not in [429, 500, 502, 503, 504] or attempt == 3:
                raise
            print('Retrying scope registration after HTTP', error.code, flush=True)
            time.sleep(2 ** attempt)
    cases = json.loads((ROOT / 'cases.json').read_text())
    actions = sorted({case['action'] for case in cases})[:args.actions]
    cases = [case for case in cases if case['action'] in actions]
    models = args.models or MODELS
    assert set(models) <= set(MODELS)
    spent = sum(json.loads(path.read_text()).get('usage', {}).get('cost', 0) or 0
                for folder in ['responses', 'pilot-mcq-responses']
                for path in (ROOT / folder).glob('*/*/*.json'))
    lock = threading.Lock()
    stopped = threading.Event()
    reserved = 0.0

    def run_cell(cell):
        nonlocal spent, reserved
        case, answer_format, model = cell
        with lock:
            if stopped.is_set():
                return
            directory = ROOT / 'responses' / model.replace('/', '__') / case['sample_id']
            output = directory / f'{answer_format}.json'
            if output.exists():
                saved = json.loads(output.read_text())
                choice = saved['choices'][0]
                if not choice['message'].get('content') or choice.get('finish_reason') != 'stop':
                    stopped.set()
                    raise RuntimeError(f'Existing response is incomplete: {output}')
                return
            if spent + reserved + 0.5 > args.budget:
                raise RuntimeError(f'Spend limit approached: ${spent:.4f}')
            reserved += 0.5
        try:
            directory = ROOT / 'responses' / model.replace('/', '__') / case['sample_id']
            output = directory / f'{answer_format}.json'
            body = request_body(case, model, answer_format)
            body_bytes = json.dumps(body).encode()
            directory.mkdir(parents=True, exist_ok=True)
            request_file = directory / f'{answer_format}.request.json'
            write_json(request_file, body)
            started = time.monotonic()
            request = urllib.request.Request(GATEWAY + '/openrouter/chat/completions',
                                             data=body_bytes, headers=headers)
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(request, timeout=240) as response:
                        result = json.load(response)
                    break
                except urllib.error.HTTPError as error:
                    error_body = error.read().decode(errors='replace').replace(key, '[redacted]')
                    write_json(directory / f'{answer_format}.error-{attempt}.json', {
                        'status': error.code, 'body': error_body,
                        'model': model, 'sample_id': case['sample_id'], 'format': answer_format})
                    safe_retry = error.code == 429
                    try:
                        safe_retry |= json.loads(error_body).get('retry_safe') is True
                    except ValueError:
                        pass
                    if not safe_retry or attempt == 3:
                        raise RuntimeError(f'Provider error {error.code}: {model}') from None
                    print('Retrying', model, case['sample_id'], 'after HTTP', error.code, flush=True)
                    time.sleep(2 ** attempt)
            result['elapsed_seconds'] = time.monotonic() - started
            result['request_sha256'] = digest(body_bytes)
            write_json(output, result)
            cost = result.get('usage', {}).get('cost')
            if cost is None:
                raise RuntimeError(f'Missing cost accounting: {output}')
            with lock:
                spent += cost
            choice = result['choices'][0]
            text = choice['message'].get('content')
            print(model, case['sample_id'], answer_format, 'answer:', repr(text),
                  'finish:', choice.get('finish_reason'), 'cost:', cost, flush=True)
            if not text or choice.get('finish_reason') != 'stop':
                raise RuntimeError(f'Incomplete answer: {output}')
        except Exception:
            stopped.set()
            raise
        finally:
            with lock:
                reserved -= 0.5

    cells = [(case, answer_format, model) for case in cases
             for answer_format in ['mcq', 'freeform'] for model in models]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _ in pool.map(run_cell, cells):
            pass


def score(args):
    import numpy as np
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    scorer_id = 'sentence-transformers/all-MiniLM-L6-v2'
    if (ROOT / 'scorer.json').exists():
        revision = json.loads((ROOT / 'scorer.json').read_text())['revision']
    else:
        revision = json.loads(fetch('https://huggingface.co/api/models/' + scorer_id))['sha']
    tokenizer_path = hf_hub_download(scorer_id, 'tokenizer.json', revision=revision)
    weights = hf_hub_download(scorer_id, 'onnx/model.onnx', revision=revision)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(max_length=256)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(weights, options, providers=['CPUExecutionProvider'])

    def embed(text):
        encoding = tokenizer.encode(text)
        inputs = {'input_ids': np.array([encoding.ids], dtype=np.int64),
                  'attention_mask': np.array([encoding.attention_mask], dtype=np.int64),
                  'token_type_ids': np.array([encoding.type_ids], dtype=np.int64)}
        hidden = session.run(None, {item.name: inputs[item.name] for item in session.get_inputs()})[0]
        mask = inputs['attention_mask'][..., None]
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        return (pooled / np.linalg.norm(pooled, axis=1, keepdims=True))[0]

    assert abs(float(embed('shooting arrow') @ embed('shooting arrow')) - 1) < 1e-5
    cases = json.loads((ROOT / 'cases.json').read_text())
    rows = []
    for model in args.models or MODELS:
        for case in cases:
            for answer_format in ['mcq', 'freeform']:
                path = ROOT / 'responses' / model.replace('/', '__') / case['sample_id'] / f'{answer_format}.json'
                if not path.exists():
                    if args.partial:
                        continue
                    raise SystemExit(f'Missing result: {path}')
                result = json.loads(path.read_text())
                choice = result['choices'][0]
                text = choice['message'].get('content') or ''
                valid = bool(text.strip()) and choice.get('finish_reason') == 'stop'
                similarity = None
                if answer_format == 'mcq':
                    answer = text.strip().strip('.').upper()
                    valid = valid and answer in 'ABCD' and len(answer) == 1
                    correct = valid and answer == case['correct_letter']
                else:
                    lines = [line.strip() for line in text.splitlines() if line.strip()]
                    answer = lines[-1].replace('Answer:', '').strip() if lines else ''
                    similarity = max(float(embed(answer) @ embed(label))
                                     for label in case['label'].split('|')) if valid else None
                    correct = valid and similarity >= 0.5
                rows.append({'model': model, 'sample_id': case['sample_id'], 'action': case['action'],
                    'variant': case['config_name'], 'format': answer_format, 'answer': answer,
                    'label': case['label'], 'valid': valid, 'correct': correct,
                    'similarity': similarity, 'cost': result['usage']['cost'],
                    'prompt_tokens': result['usage']['prompt_tokens'],
                    'completion_tokens': result['usage']['completion_tokens'],
                    'elapsed_seconds': result['elapsed_seconds']})
    write_json(ROOT / 'scores.json', rows)
    write_json(ROOT / 'scorer.json', {'model': scorer_id, 'revision': revision,
               'weights_sha256': digest(Path(weights).read_bytes()), 'threshold': 0.5,
               'pooling': 'attention-masked mean followed by L2 normalization'})
    groups = []
    for model in args.models or MODELS:
        for variant in VARIANTS:
            for answer_format in ['mcq', 'freeform']:
                group = [row for row in rows if row['model'] == model and row['variant'] == variant
                         and row['format'] == answer_format]
                if not group:
                    continue
                n = len(group)
                rate = sum(row['correct'] for row in group) / n
                denominator = 1 + 1.96 ** 2 / n
                center = (rate + 1.96 ** 2 / (2 * n)) / denominator
                radius = 1.96 * math.sqrt(rate * (1 - rate) / n + 1.96 ** 2 / (4 * n ** 2)) / denominator
                groups.append({'model': model, 'variant': variant, 'format': answer_format,
                    'n': n, 'correct': sum(row['correct'] for row in group),
                    'invalid': sum(not row['valid'] for row in group),
                    'accuracy': rate, 'wilson_95_low': center - radius, 'wilson_95_high': center + radius,
                    'cost': sum(row['cost'] for row in group)})
    write_json(ROOT / 'summary.json', groups)
    print(json.dumps(groups, indent=2))


def verify(args):
    cases = json.loads((ROOT / 'cases.json').read_text())
    rows = json.loads((ROOT / 'scores.json').read_text())
    summary = json.loads((ROOT / 'summary.json').read_text())
    assert len(cases) == 172
    expected = {(model, case['sample_id'], fmt) for model in MODELS
                for case in cases for fmt in ['mcq', 'freeform']}
    observed = {(row['model'], row['sample_id'], row['format']) for row in rows}
    assert observed == expected and len(rows) == len(expected) == 2064
    assert all(row['valid'] for row in rows)
    total_cost = 0
    for case in cases:
        assert case['options'][ord(case['correct_letter']) - 65] in case['label'].split('|')
        directory = ROOT / 'cases' / case['sample_id']
        assert digest((directory / 'video.mp4').read_bytes()) == case['video_sha256']
        for model in MODELS:
            for fmt in ['mcq', 'freeform']:
                directory = ROOT / 'responses' / model.replace('/', '__') / case['sample_id']
                request = json.loads((directory / f'{fmt}.request.json').read_text())
                response = json.loads((directory / f'{fmt}.json').read_text())
                assert request == request_body(case, model, fmt)
                assert digest(json.dumps(request).encode()) == response['request_sha256']
                assert response['model'] == model
                assert response['choices'][0]['finish_reason'] == 'stop'
                total_cost += response['usage']['cost']
    for group in summary:
        selected = [row for row in rows if all(row[k] == group[k] for k in ['model', 'variant', 'format'])]
        assert len(selected) == group['n'] == 86
        assert sum(row['correct'] for row in selected) == group['correct']
    assert math.isclose(total_cost, sum(group['cost'] for group in summary))
    validation_cost = sum(json.loads(path.read_text()).get('usage', {}).get('cost', 0) or 0
                          for path in (ROOT / 'pilot-mcq-responses').glob('*/*/*.json'))
    report = {'passed': True, 'predictions': len(rows), 'videos': len(cases),
              'models': len(MODELS), 'invalid': 0, 'model_charges_usd': total_cost,
              'validation_charges_usd': validation_cost, 'total_charges_usd': total_cost + validation_cost,
              'checks': ['coverage', 'request provenance', 'image hashes', 'model identities',
                         'completed outputs', 'score aggregation', 'cost reconciliation']}
    write_json(ROOT / 'integrity-report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'predict', 'score', 'verify'])
    parser.add_argument('--actions', type=int, default=86)
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--budget', type=float, default=60)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    {'prepare': prepare, 'predict': predict, 'score': score, 'verify': verify}[args.command](args)
