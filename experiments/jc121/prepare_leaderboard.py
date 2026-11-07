"""Prepare archived and current MIME answers for a common Jev grading rubric."""

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from judge_jev import CRITERIA, INSTRUCTIONS, RESOLVED_MODEL, save

REPO = Path(__file__).resolve().parents[2]
RUN = Path('/mnt/efs/justin/0_experiments/jc121')
VARIANTS = ('base+blank@0', 'adversarial+misaligned@0')
NAMES = {
    'gemini_gemini-1.5-flash': 'Gemini 1.5 Flash',
    'gemini-few-shot': 'Gemini 1.5 Flash',
    'openai_gpt-4o-mini': 'GPT-4o Mini',
    'openai-few-shot': 'GPT-4o Mini',
    'internvl2_5_OpenGVLab_InternVL2_5-8B': 'InternVL2.5 (8B)',
    'phi35_microsoft_Phi-3.5-vision-instruct': 'Phi 3.5 (4.2B)',
    'qwen25vl_Qwen_Qwen2.5-VL-7B-Instruct': 'Qwen 2.5 VL (7B)',
    'qwen25vl_Qwen_Qwen2.5-VL-3B-Instruct': 'Qwen 2.5 VL (3B)',
    'qwen2vl_Qwen_Qwen2-VL-7B-Instruct': 'Qwen 2 VL (7B)',
}
EVALUATED_MODELS = {
    'openai/gpt-6-astra': 'GPT-6 Astra',
    'openai/gpt-6.1-sol': 'GPT-6.1 Sol',
    'anthropic/claude-opus-5.5': 'Claude Opus 5.5',
    'deepseek/deepseek-v4.1-flash-1017': 'DeepSeek V4.1 Flash 1017',
    'qwen/qwen3.8-flash': 'Qwen3.8 Flash',
    'z-ai/glm-5.3-flash': 'GLM 5.3 Flash',
    'minimax/minimax-m3.1-flash': 'MiniMax M3.1 Flash',
}


def canonical_file(value):
    return Path(value).name.replace('_Justin_', '_j_').replace('_Natali_', '_n_').lower()


def prepare(root):
    cases = json.loads((RUN / 'cases.json').read_text())
    by_file = {canonical_file(c['file_name']): c for c in cases}
    by_id = {c['sample_id']: c for c in cases}
    legacy_path = REPO / 'data/mime_data_legacy.jsonl'
    legacy = [json.loads(line) for line in legacy_path.read_text().splitlines()]
    rows, pairs, hashes = [], {}, {}

    def track(path):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()

    def add(record, case, answer=None):
        record.update(sample_id=case['sample_id'], action=case['action'], variant=case['config_name'])
        if answer is not None:
            refs = case['label'].split('|')
            encoded = json.dumps({'references': refs, 'answer': answer}, sort_keys=True).encode()
            pair_id = hashlib.sha256(encoded).hexdigest()[:12]
            pair = {'id': pair_id, 'references': refs, 'answer': answer}
            assert pair_id not in pairs or pairs[pair_id] == pair
            pairs[pair_id] = pair
            record['pair_id'] = pair_id
        rows.append(record)

    track(legacy_path)
    track(RUN / 'cases.json')
    for path in sorted((REPO / 'results/mime-cropped').glob('*/all/score_*.csv')):
        method = path.parts[-3]
        fmt, model = path.stem.removeprefix('score_').split('_', 1)
        track(path)
        selected = []
        for row in csv.DictReader(path.open()):
            case = by_file.get(canonical_file(legacy[int(row['instance_id'])]['file_name']))
            if case is None:
                continue
            record = {'model': NAMES[model], 'method': method, 'format': fmt,
                      'cohort': '2025', 'source': str(path.relative_to(REPO)),
                      'source_row': int(row['instance_id'])}
            if fmt == 'ff':
                assert ast.literal_eval(row['ground_truth']) == case['label'].split('|'), path
                record['embedding_credit'] = row['correct'].lower() == 'true'
                add(record, case, row['output_text'])
            else:
                record['credit'] = row.get('is_correct', row.get('correct')).lower() in ('true', '1')
                add(record, case)
            selected.append(case['sample_id'])
        assert len(selected) == len(set(selected)) == 172, (path, len(selected))

    human_path = REPO / 'results/mime-cropped/human_eval.jsonl'
    track(human_path)
    seen = set()
    for index, line in enumerate(human_path.read_text().splitlines()):
        row = json.loads(line)
        dedup = (row['url_data']['annotator'], row['question']['sample_id'])
        if dedup in seen:
            continue
        seen.add(dedup)
        case = by_file.get(canonical_file(row['question']['s3_url']))
        if case is None:
            continue
        assert row['question']['label'] == case['label'].split('|'), index
        record = {'model': 'Human', 'method': 'human', 'cohort': 'human',
                  'source': str(human_path.relative_to(REPO)), 'source_row': index}
        add(dict(record, format='ff'), case, row['user_answer'])
        credit = str(row['user_is_correct']).lower() == 'true'
        add(dict(record, format='mcq', credit=credit), case)

    current_path = RUN / 'studies/jev-judge/joined-jev.json'
    track(current_path)
    for row in json.loads(current_path.read_text()):
        add({'model': EVALUATED_MODELS[row['model']], 'model_id': row['model'], 'method': 'zero-shot',
             'format': 'ff', 'cohort': '2026', 'source': str(current_path),
             'embedding_credit': row['correct']}, by_id[row['sample_id']], row['answer'])
    scores_path = RUN / 'revisions/jc121.1/scores.json'
    track(scores_path)
    scores = json.loads(scores_path.read_text())
    for row in scores:
        if row['format'] != 'mcq':
            continue
        add({'model': EVALUATED_MODELS[row['model']], 'model_id': row['model'], 'method': 'zero-shot',
             'format': 'mcq', 'cohort': '2026', 'source': str(scores_path),
             'credit': row['correct']}, by_id[row['sample_id']])

    save(root / 'observations.json', rows)
    save(root / 'blind-pairs.json', list(pairs.values()))
    save(root / 'contract.json', {
        'purpose': 'Update the MIME leaderboard with a common free-form grader for archived models, humans, and current models.',
        'judge': RESOLVED_MODEL, 'instructions': INSTRUCTIONS, 'criteria': CRITERIA,
        'threshold': 0.5, 'source_sha256': hashes, 'variants': VARIANTS,
        'dataset_revision': cases[0]['dataset_revision'],
        'historical_answers': 'Published score CSV output_text, joined by instance_id to frozen legacy metadata. Raw prediction archives can differ from published score CSVs.',
        'human_sampling': 'Deduplicate annotator and question sample_id; preserve all available responses and report answer and video counts.',
        'limitations': 'Different model generations and prompting methods, recorded in each row. Historical model generation was not rerun. Human video presentation differs from sampled model frames.',
        'additional_api_budget_usd': 1,
    })
    cached = {}
    cache_dir = RUN / 'studies/jev-judge/responses'
    for request_path in cache_dir.glob('*.request.json'):
        response_path = request_path.with_name(request_path.name.replace('.request.', '.response.'))
        if response_path.exists():
            body = json.loads(request_path.read_text())
            key = json.dumps(body['state'], sort_keys=True)
            cached[key] = (body, json.loads(response_path.read_text()))
    reused = 0
    for pair in pairs.values():
        key = json.dumps({'reference_actions': pair['references'], 'candidate_answer': pair['answer']}, sort_keys=True)
        if key in cached:
            body, response = cached[key]
            save(root / 'responses' / (pair['id'] + '.request.json'), body)
            save(root / 'responses' / (pair['id'] + '.response.json'), response)
            reused += 1
    save(root / 'preparation.json', {'observations': len(rows), 'pairs': len(pairs),
                                   'reused_pairs': reused, 'new_pairs': len(pairs) - reused,
                                   'cohort_counts': dict(Counter(r['cohort'] for r in rows))})
    print((root / 'preparation.json').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output_dir)
