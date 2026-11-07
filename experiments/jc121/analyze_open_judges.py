"""Compare saved open-judge predictions with the frozen MIME audit."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from judge_jev import CRITERIA, INSTRUCTIONS


def counts(observations):
    tp = sum(pred and truth for pred, truth in observations)
    fp = sum(pred and not truth for pred, truth in observations)
    fn = sum(not pred and truth for pred, truth in observations)
    tn = sum(not pred and not truth for pred, truth in observations)
    n = tp + fp + fn + tn
    return {'answers': n, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'agreement_pct': 100 * (tp + tn) / n,
            'precision_pct': 100 * tp / (tp + fp) if tp + fp else None,
            'recall_pct': 100 * tp / (tp + fn),
            'balanced_accuracy_pct': 50 * (tp / (tp + fn) + tn / (tn + fp))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    root = args.root
    contract = json.loads((root / 'contract.json').read_text())
    for file, expected in contract['source_hashes'].items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == expected, file
    rows = json.loads((root / 'joined.json').read_text())
    pairs = {p['id']: p for p in json.loads((root / 'blind-pairs.json').read_text())}
    definite = [r for r in rows if r['audit_verdict'] in ['match', 'mismatch']]
    assert len(pairs) == 840 and len(rows) == 1204 and len(definite) == 1127
    predictions = {'embedding': {r['pair_id']: r['correct'] for r in rows},
                   'jev': {k: v['credit'] for k, v in json.loads((root / 'jev-predictions.json').read_text()).items()}}
    resources = {}
    outputs = {}
    for name in contract['models']:
        files = list((root / name / 'responses').glob('*.json'))
        if args.partial and len(files) != 840:
            continue
        assert len(files) == 840, (name, len(files))
        predictions[name], outputs[name] = {}, {}
        for file in files:
            d = json.loads(file.read_text())
            pair = pairs[d['pair_id']]
            expected_request = {'state': {'reference_actions': pair['references'], 'candidate_answer': pair['answer']},
                                'questions': {'correct': {'type': 'noul', 'instructions': INSTRUCTIONS, 'criteria': CRITERIA}}}
            assert d['request'] == expected_request
            assert d['request_sha256'] == hashlib.sha256(json.dumps(expected_request, sort_keys=True).encode()).hexdigest()
            assert d['credit'] == (d['probability'] >= 0.5)
            assert 0 <= d['probability'] <= 1
            assert d['model_revision'] == json.loads((root / 'models.json').read_text())[name]['revision']
            predictions[name][d['pair_id']] = d['credit']
            outputs[name][d['pair_id']] = d
        assert set(predictions[name]) == set(pairs)
        timing_path = root / name / 'timing.json'
        resources[name] = json.loads(timing_path.read_text()) if timing_path.exists() else {}
        resources[name]['total_pair_inference_seconds'] = sum(d['elapsed_seconds'] for d in outputs[name].values())
        if name == 'verdict':
            resources[name]['abstained_pairs'] = sum(d['response']['results'][0]['is_abstention'] for d in outputs[name].values())
        if 'seconds_per_pair' in resources[name]:
            values = resources[name]['seconds_per_pair']
            resources[name].update(median_seconds_per_pair=statistics.median(values),
                                   min_seconds_per_pair=min(values), max_seconds_per_pair=max(values))
    metrics, model_scores, disagreements = [], [], []
    unique_definite = {r['pair_id']: r['audit_verdict'] == 'match' for r in definite}
    for name, pred in predictions.items():
        summary = {'grader': name, **counts([(pred[r['pair_id']], r['audit_verdict'] == 'match') for r in definite])}
        summary['unique_pairs'] = counts([(pred[uid], truth) for uid, truth in unique_definite.items()])
        for model in sorted({r['model'] for r in rows}):
            for variant in ['base+blank@0', 'adversarial+misaligned@0']:
                group = [r for r in rows if r['model'] == model and r['variant'] == variant]
                assert len(group) == 86
                predicted = 100 * sum(pred[r['pair_id']] for r in group) / 86
                adjusted = 100 * sum(r['audit_verdict'] == 'match' if r['audit_verdict'] in ['match','mismatch'] else r['correct'] for r in group) / 86
                model_scores.append({'grader': name, 'model': model, 'variant': variant,
                                     'accuracy_pct': predicted, 'audit_adjusted_pct': adjusted,
                                     'absolute_gap_pp': abs(predicted - adjusted)})
        summary['mean_absolute_model_score_gap_pp'] = statistics.mean(r['absolute_gap_pp'] for r in model_scores if r['grader'] == name)
        original_wrong = [r for r in definite if r['correct'] != (r['audit_verdict'] == 'match')]
        summary['embedding_errors_fixed'] = sum(pred[r['pair_id']] == (r['audit_verdict'] == 'match') for r in original_wrong)
        summary['new_errors_vs_embedding'] = sum(pred[r['pair_id']] != (r['audit_verdict'] == 'match') for r in definite if r not in original_wrong)
        summary['runtime'] = resources.get(name)
        metrics.append(summary)
        for row in definite:
            if pred[row['pair_id']] != (row['audit_verdict'] == 'match'):
                disagreements.append({'grader': name, 'model': row['model'], 'variant': row['variant'],
                     'reference': ' | '.join(row['references']), 'answer': row['answer'],
                     'audit': row['audit_verdict'], 'predicted_credit': pred[row['pair_id']],
                     'probability': outputs.get(name, {}).get(row['pair_id'], {}).get('probability')})
    complete = all(name in predictions for name in contract['models'])
    report = {'complete': complete, 'metrics': metrics, 'resources': resources,
              'pairs': 840, 'answers': 1204, 'definite_answers':1127, 'ambiguous_answers':77,
              'limitation':'Agreement with an AI-assisted audit, not independent human ground truth.'}
    (root / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    for name, records in [('model-scores.csv', model_scores), ('disagreements.csv', disagreements)]:
        with (root / name).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    (root / 'integrity-report.json').write_text(json.dumps({'passed':complete,'source_hashes_verified':True,
          'coverage':{name:len(p) for name,p in predictions.items()},'blinded_requests_verified':True,
          'threshold':0.5,'no_audit_training_or_tuning':True},indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
