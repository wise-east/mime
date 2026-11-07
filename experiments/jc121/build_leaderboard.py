"""Verify saved judgments and export the MIME leaderboard and sharing figure."""

import argparse
from collections import defaultdict
import csv
import hashlib
from html import escape
import json
import math
from pathlib import Path
import re

from judge_jev import CRITERIA, INSTRUCTIONS, RESOLVED_MODEL, probability, save


def build(root, website):
    contract = json.loads((root / 'contract.json').read_text())
    for path, digest in contract['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    pairs = json.loads((root / 'blind-pairs.json').read_text())
    predictions, total_cost = {}, 0
    for pair in pairs:
        stem = root / 'responses' / pair['id']
        body = json.loads(stem.with_suffix('.request.json').read_text())
        response = json.loads(stem.with_suffix('.response.json').read_text())
        assert body['state'] == {'reference_actions': pair['references'], 'candidate_answer': pair['answer']}
        assert body['questions'] == {'correct': {'type': 'noul', 'instructions': INSTRUCTIONS, 'criteria': CRITERIA}}
        assert response['request_sha256'] == hashlib.sha256(json.dumps(body).encode()).hexdigest()
        predictions[pair['id']] = probability(response) >= 0.5
        total_cost += response['usage']['cost']
    observations = json.loads((root / 'observations.json').read_text())
    groups = defaultdict(list)
    for row in observations:
        row['credit'] = predictions[row['pair_id']] if row['format'] == 'ff' else row['credit']
        groups[(row['model'], row['method'], row['cohort'], row['variant'], row['format'])].append(row)
    results = []
    for (model, method, cohort, variant, fmt), rows in sorted(groups.items()):
        videos = len({r['sample_id'] for r in rows})
        if cohort != 'human':
            assert len(rows) == videos == 86
        else:
            assert (len(rows), videos) == (258, 86)
        correct = sum(r['credit'] for r in rows)
        result = dict(model=model, method=method, cohort=cohort, variant=variant, format=fmt,
                      correct=correct, answers=len(rows), videos=videos, accuracy_pct=100 * correct / len(rows))
        if fmt == 'ff' and cohort != 'human':
            result['embedding_pct'] = 100 * sum(r['embedding_credit'] for r in rows) / len(rows)
        results.append(result)
    preparation = json.loads((root / 'preparation.json').read_text())
    reused_cost = json.loads((root.parent / 'jev-judge/report.json').read_text())['usage']['cost_usd']
    data = {'date': '2026-11-07', 'judge': RESOLVED_MODEL, 'threshold': 0.5,
            'dataset_revision': contract['dataset_revision'],
            'accuracy_definition': 'Percentage of correct answers, from 0 to 100; higher is better. Free-form uses Jev semantic judgments, multiple choice uses the correct option.',
            'conditions': list(contract['variants']), 'results': results,
            'method_notes': [
                'Current models: zero-shot, medium reasoning requested; Qwen3.8 Flash uses a 4096-token thinking budget. Every 30th video frame, starting at frame 29, resized within 512 by 512 pixels.',
                'Archived 2025 answers are rescored, not regenerated. Historical prompting and model versions differ from the current runs.',
                'All free-form answers, including human responses, use Jev 1.14 at probability >= 0.5. Historical embedding scores remain in the archive.',
                'Humans viewed videos and supplied 3 answers for each of 86 videos per condition after deduplication. Models received sampled frames for the same 86 videos.',
                'Human free-form scores use Jev and can undercount shorthand answers. The original human free-form calculation counted some low-similarity mismatches as correct and is preserved only as a historical result.',
                'The Jev validation measured agreement with an AI-assisted audit, not independent human-labeled accuracy.',
            ],
            'sources': {'code': 'https://github.com/wise-east/mime/tree/codex/mime-leaderboard',
                        'data': 'https://huggingface.co/datasets/wise-east/mime-cropped'},
            'grading_cost': {'all_unique_pairs_usd': total_cost, 'reused_usd': reused_cost,
                             'additional_usd': total_cost - reused_cost,
                             'scope': 'Jev grading API charges only. Excludes answer generation, shared infrastructure, and working time.'}}
    save(root / 'leaderboard.json', data)
    with (root / 'leaderboard.csv').open('w') as f:
        fields = ['model', 'method', 'cohort', 'variant', 'format', 'correct', 'answers', 'videos', 'accuracy_pct', 'embedding_pct']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    save(root / 'integrity-report.json', {'passed': True, 'judged_pairs': len(predictions),
                                        'result_rows': len(results), 'observations': len(observations),
                                        'new_pairs': preparation['new_pairs'],
                                        'missing_responses': 0, 'threshold': 0.5})
    assets = website / 'assets/mime'
    save(assets / 'leaderboard.json', data)
    page_path = website / '_research/mime.html'
    page = page_path.read_text()
    methods = {'zero-shot': 'Zero-shot', 'cot': 'CoT', 'few-shot': 'Few-shot', 'human': 'Human'}
    tables = [('ff-adv', 'adversarial+misaligned@0', 'ff'), ('mc-adv', 'adversarial+misaligned@0', 'mcq'),
              ('ff-base', 'base+blank@0', 'ff'), ('mc-base', 'base+blank@0', 'mcq')]
    for table_id, variant, fmt in tables:
        selected = [r for r in results if r['variant'] == variant and r['format'] == fmt]
        selected.sort(key=lambda r: (r['cohort'] != 'human', -r['accuracy_pct'], r['model']))
        rendered, rank = [], 0
        for r in selected:
            human = r['cohort'] == 'human'
            if not human:
                rank += 1
            name = 'Archived 2025 answers' if r['cohort'] in ('human', '2025') else 'MIME update 2026'
            method = methods[r['method']]
            if r['cohort'] == '2026':
                method += ', medium reasoning' if 'Qwen3.8' not in r['model'] else ', 4096-token thinking'
            if fmt == 'ff':
                method += ' / Jev'
            date = '2025-07-03' if r['cohort'] in ('human', '2025') else data['date']
            cls = ' class="human-row"' if human else ''
            cells = ['-' if human else str(rank), name, r['model'], method, f"{r['accuracy_pct']:.1f}", date]
            rendered.append('                  <tr' + cls + '>' + ''.join('<td>' + escape(c) + '</td>' for c in cells) + '</tr>')
        pattern = rf'(<div class="mime-leaderboard-table[^\"]*" id="mime-table-{table_id}">.*?<tbody>).*?(</tbody>)'
        page, n = re.subn(pattern, lambda m: m[1] + '\n' + '\n'.join(rendered) + '\n                ' + m[2], page, count=1, flags=re.S)
        assert n == 1, table_id
    notes = '''      <div id="mime-scoring-notes" class="content" style="max-width:1000px;margin:1.5rem auto;padding:0 1rem;">
        <p><strong>November 2026 update.</strong> Accuracy is the percentage of correct answers, from 0 to 100; higher is better. Free-form answers from every model and humans are graded by Jev 1.14 using the same rubric and a 0.5 probability threshold. Multiple-choice answers use the correct option.</p>
        <p>Each model answers 86 videos per condition. Humans viewed videos rather than sampled frames and supplied three responses for each of the same 86 videos. Current models use zero-shot prompts with medium reasoning requested; Qwen3.8 Flash uses a 4096-token thinking budget. Historical answers retain their original prompting methods and are rescored without regenerating them.</p>
        <p>The 2025 dates identify the archived answers; their free-form scores here use the current grader. Some archived answer files differ from the old website scores. The original published tables are preserved below. Jev agreed with 99.7% of 1,127 definite AI-assisted audit labels; that audit is not independent human ground truth.</p>
        <p>Human free-form scores here are Jev-scored responses, not independently verified human accuracy. Jev can reject shorthand, such as “violin” for “playing violin.” The original human free-form calculation also counted some low-similarity mismatches as correct, so its published values are not a reliable accuracy reference. Human multiple-choice scores do not use that calculation.</p>
        <p><a href="/assets/mime/leaderboard.json">Download scores and methods</a> · <a href="/assets/mime/progress.png">Download the figure</a> · <a href="https://github.com/wise-east/mime/tree/codex/mime-leaderboard">Evaluation code</a></p>
        <figure><img src="/assets/mime/progress.png" alt="MIME action recognition accuracy for archived and current models, compared with human responses, under base and adversarial conditions." style="width:100%;height:auto;" /></figure>
      </div>
'''
    if 'id="mime-scoring-notes"' in page:
        page = re.sub(r'      <div id="mime-scoring-notes".*?</div>\n', notes, page, count=1, flags=re.S)
    else:
        page = page.replace('      <div class="mime-tabs">', notes + '\n      <div class="mime-tabs">', 1)
    if 'mime-published-archive' not in page:
        archive = '''      <details id="mime-published-archive" class="content" style="margin:2rem auto;max-width:1000px;">
        <summary>Original published 2025 leaderboard</summary>
        <p>These tables preserve the original website scores and embedding-based free-form grading. Their scores should not be mixed with the Jev-regraded tables above. The original human free-form calculation counted some low-similarity mismatches as correct.</p>
        {% include mime-leaderboard-2025.html %}
      </details>
'''
        marker = '      <div class="columns is-centered has-text-centered">'
        start = page.index('      <div id="mime-leaderboard-tables">')
        end = page.index(marker, start)
        page = page[:end] + archive + '\n' + page[end:]
    page_path.write_text(page)
    figure(results, assets / 'progress.svg')
    print(json.dumps(data['grading_cost'], indent=2))


def figure(results, path):
    plotted = []
    selected = [None, 'GLM 5.3 Flash', 'Claude Opus 5.5', 'GPT-6.1 Sol', 'GPT-6 Astra']
    labels = [['Best 2025', 'result'], ['GLM 5.3', 'Flash'], ['Claude', 'Opus 5.5'], ['GPT-6.1', 'Sol'], ['GPT-6', 'Astra']]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1000" viewBox="0 0 1200 1000">',
           '<rect width="1200" height="1000" fill="white"/>',
           '<style>text{font-family:"DejaVu Sans",sans-serif;fill:#263345} .muted{fill:#64748b}</style>',
           '<text x="600" y="47" text-anchor="middle" font-size="31">Progress on MIME, with a free-form gap</text>']
    def text(x, y, content, size=18, color=None, anchor='start'):
        style = f' style="fill:{color}"' if color else ''
        svg.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}"{style}>{escape(str(content))}</text>')
    for col, variant in enumerate(['base+blank@0', 'adversarial+misaligned@0']):
        left, width = 90 + 585 * col, 465
        text(left + width/2, 95, 'Blank background' if col == 0 else 'Adversarial avatar + background', 23, anchor='middle')
        for row, fmt in enumerate(['ff', 'mcq']):
            top, height = 155 + 405 * row, 275
            bottom = top + height
            text(left + width/2, top - 20, 'Free-form' if fmt == 'ff' else 'Multiple choice', 22, anchor='middle')
            for tick in [0, 25, 50, 75, 100]:
                y = bottom - height*tick/100
                svg.append(f'<line x1="{left}" y1="{y}" x2="{left+width}" y2="{y}" stroke="#e6eaf0"/>')
                text(left-12, y+6, tick, 16, '#64748b', 'end')
            svg.append(f'<path d="M {left} {top} V {bottom} H {left+width}" fill="none" stroke="#64748b"/>')
            if col == 0:
                svg.append(f'<text transform="translate({left-57},{top+height/2}) rotate(-90)" text-anchor="middle" font-size="18">Accuracy (%)</text>')
            human = next(r for r in results if r['cohort'] == 'human' and r['variant'] == variant and r['format'] == fmt)
            plotted.append(human)
            hy = bottom - height*human['accuracy_pct']/100
            svg.append(f'<line x1="{left}" y1="{hy}" x2="{left+width}" y2="{hy}" stroke="#b7791f" stroke-width="2.2" stroke-dasharray="7 5"/>')
            human_label = 'Human · Jev' if fmt == 'ff' else 'Human'
            text(left+8, hy+23, f"{human_label} {human['accuracy_pct']:.1f}%", 17, '#9c6417')
            for i, model in enumerate(selected):
                if model is None:
                    r = max((r for r in results if r['cohort'] == '2025' and r['variant'] == variant and r['format'] == fmt), key=lambda r: r['accuracy_pct'])
                else:
                    r = next(r for r in results if r['model'] == model and r['method'] == 'zero-shot' and r['variant'] == variant and r['format'] == fmt)
                plotted.append(r)
                x = left + 15 + i*91
                barh = height*r['accuracy_pct']/100
                color = '#a8afb9' if i == 0 else ('#167d8d' if i == 4 else '#abcbd0')
                svg.append(f'<rect x="{x}" y="{bottom-barh}" width="62" height="{barh}" fill="{color}"/>')
                if i == 4:
                    if fmt == 'mcq':
                        svg.append(f'<rect x="{x+3}" y="{bottom-barh-32}" width="56" height="25" fill="white"/>')
                    text(x+31, bottom-barh-9, f"{r['accuracy_pct']:.1f}", 19, '#126776', 'middle')
                for j, label in enumerate(labels[i]):
                    text(x+31, bottom+27+22*j, label, 17, anchor='middle')
    text(600, 931, '86 videos per condition · Three human responses per video · Jev 1.14 free-form scoring', 17, '#64748b', 'middle')
    text(600, 959, '2025: best across methods · 2026: zero-shot, medium reasoning requested · Higher is better', 16, '#64748b', 'middle')
    text(600, 987, 'justin-cho.com/mime · November 2026', 17, '#64748b', 'middle')
    svg.append('</svg>')
    path.write_text('\n'.join(svg))
    save(path.with_name('progress-data.json'), plotted)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--website', type=Path, required=True)
    args = parser.parse_args()
    build(args.output_dir, args.website)
