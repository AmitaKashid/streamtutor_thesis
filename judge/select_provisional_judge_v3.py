"""Post-hoc provisional scalar judge decision support for frozen v3 evidence.

Usage: python -m judge.select_provisional_judge_v3 --input-zip day4_task2_frozen_inputs.zip --output-dir judge_outputs/day4_selection
No network/GPU. Does not modify evidence. Output cannot replace expert validation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

MODELS = {
    'glm4_32b_0414_q4': 'GLM 32B',
    'compassjudger_2_7b_q4': 'CompassJudger 7B',
    'fare_20b_q4': 'FARE 20B',
    'prometheus_7b_v2_q4': 'Prometheus 7B',
}
SPLITS = {'calibration':132, 'selection':792, 'robustness':123, 'contrasts':28}
CRITICAL_DIMENSIONS = ('correctness','groundedness','conversational_context')


def analyze(archive: Path) -> dict:
    by_model = defaultdict(dict)
    pair_ids = {}
    with ZipFile(archive) as z:
        for fn in z.namelist():
            if not fn.endswith('/manifest.json'):
                continue
            folder = fn.rsplit('/',1)[0]
            manifest = json.loads(z.read(fn))
            model = manifest['judge_configuration']['model_id']
            split = manifest.get('split', 'contrasts' if 'contrasts' in folder else None)
            if model not in MODELS or split not in SPLITS:
                raise ValueError(f'Unknown model/split in {fn}: {model}/{split}')
            if split in by_model[model]:
                raise ValueError(f'Duplicate model/split: {model}/{split}')
            pred_file = folder + ('/pairwise_predictions.jsonl' if split=='contrasts' else '/predictions.jsonl')
            records = [json.loads(line) for line in z.read(pred_file).decode().splitlines() if line.strip()]
            if len(records) != SPLITS[split] or manifest['completed_evaluation_count'] != SPLITS[split]:
                raise ValueError(f'Wrong count: {model}/{split}')
            field = 'evaluation_id'
            keys=[r[field] for r in records]
            if len(keys)!=len(set(keys)):
                raise ValueError(f'Duplicate evaluation ID: {model}/{split}')
            hashes = {r['evaluation_id']:r.get('input_sha256') for r in records if split!='contrasts'}
            if split!='contrasts':
                if split not in pair_ids:
                    pair_ids[split]=hashes
                elif hashes!=pair_ids[split]:
                    raise ValueError(f'Input hash mismatch: {model}/{split}')
            by_model[model][split] = (manifest, records)
    if set(by_model)!=set(MODELS) or any(set(by_model[m])!=set(SPLITS) for m in MODELS):
        raise ValueError('Missing model or split')
    dataset_hashes={by_model[m][s][0]['dataset']['checksums_sha256'] for m in MODELS for s in SPLITS}
    rubric_hashes={by_model[m][s][0]['dataset']['rubric_sha256'] for m in MODELS for s in SPLITS}
    if len(dataset_hashes)!=1 or len(rubric_hashes)!=1:
        raise ValueError('Dataset/rubric hash mismatch')
    out={}
    for m in MODELS:
        parts=by_model[m]
        all_rows=[r for _,recs in parts.values() for r in recs]
        selection=parts['selection'][1]
        stress=parts['robustness'][1]
        contrasts=parts['contrasts'][1]
        def scalar_group(rows):
            groups=defaultdict(dict)
            for r in rows:
                quality=(r.get('response_id') or '').split('-')[-1]
                key=(r['question_id'],r['dimension'])
                result=r['judge'].get('result') or {}
                groups[key][quality]=result.get('score') if r['judge']['status']=='valid' else None
            triples=[g for g in groups.values() if all(k in g and g[k] is not None for k in ('excellent','borderline','critical'))]
            return {'triples':len(triples),'weak_order': sum(g['excellent']>=g['borderline']>=g['critical'] for g in triples),'strict_order':sum(g['excellent']>g['borderline']>g['critical'] for g in triples), 'gap':sum(g['excellent']-g['critical'] for g in triples)/len(triples)}
        def critical_risks(dim):
            rows=[r for r in selection if r['dimension']==dim and (r.get('response_id') or '').endswith('-critical')]
            vals=[r['judge'].get('result',{}).get('score') if r['judge'].get('result') else None for r in rows]
            return {'n':len(rows),'valid':sum(x is not None for x in vals),'high_4_or_5':sum(x is not None and x>=4 for x in vals),'score_5':sum(x==5 for x in vals)}
        c_valid=sum(r['judge']['status']=='valid' for r in contrasts)
        c_pairs=defaultdict(list)
        for r in contrasts:
            c_pairs[r['contrast_id']].append(r)
        bidirectional=sum(len(v)==2 and all(r['judge']['status']=='valid' for r in v) for v in c_pairs.values())
        selection_valid=sum(r['judge']['status']=='valid' for r in selection)
        latency=[(r['judge'].get('attempts') or [{}])[-1].get('metrics',{}).get('wall_seconds') for r in selection]
        latency=sorted(t for t in latency if isinstance(t,(float,int)))
        out[m]={
            'name':MODELS[m], 'total_recorded':len(all_rows), 'valid_total':sum(r['judge']['status']=='valid' for r in all_rows),
            'selection_valid':selection_valid, 'selection_triples':scalar_group(selection),
            'stress_triples':scalar_group(stress), 'critical_risks':{d:critical_risks(d) for d in CRITICAL_DIMENSIONS},
            'contrast_valid':c_valid,'contrast_bidirectional_pairs':bidirectional,
            'selection_attempt_p95_s':latency[min(len(latency)-1,int(.95*(len(latency)-1)))] if latency else None,
        }
    # Explicitly post-hoc engineering screen. NOT a validated accuracy metric.
    eligible = []
    for model, metrics in out.items():
        triples = metrics['selection_triples']
        weak = triples['weak_order'] / triples['triples']
        reliability = metrics['valid_total'] / metrics['total_recorded']
        usable = metrics['contrast_bidirectional_pairs'] / 14
        severe = sum(metrics['critical_risks'][dim]['high_4_or_5'] for dim in ('correctness', 'groundedness'))
        if reliability >= 0.995 and weak >= 0.95 and usable >= 0.9:
            eligible.append((severe, -metrics['stress_triples']['gap'], model))
    eligible.sort()
    if not eligible:
        raise ValueError('No judge passes the provisional engineering screen')
    chosen = eligible[0][2]
    fallback = eligible[1][2] if len(eligible)>1 else None
    return {'decision_status':'provisional_post_hoc', 'selected_scalar_judge':chosen,
            'fallback_scalar_judge':fallback,
            'post_hoc_screen':{'overall_validity_min':0.995, 'weak_order_min':0.95,'bidirectional_pair_coverage_min':0.9,
                'ranking':'fewest >=4 scores on critical correctness/groundedness cases; then greater stress-set excellent-critical separation',
                'eligible_models':[e[2] for e in eligible]},
            'input_zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
            'dataset_sha256':next(iter(dataset_hashes)), 'rubric_sha256':next(iter(rubric_hashes)),
            'metrics':out,'limitations':['synthetic/provisional quality labels','only 36 independent selection question clusters','selection thresholds chosen post hoc','not validated against human ratings','pairwise preferences all favor source a','A30 deployment not verified']}


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-zip',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args(argv)
    d=analyze(args.input_zip)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'provisional_decision_metrics.json').write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
    with (args.output_dir/'critical_risk_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['model','dimension','n','valid','score_ge_4','score_5'])
        for m,row in d['metrics'].items():
            for dim,r in row['critical_risks'].items():w.writerow([m,dim,r['n'],r['valid'],r['high_4_or_5'],r['score_5']])
    print('Audited 16 runs, 4300 records; provisional scalar choice:',d['selected_scalar_judge'])

if __name__=='__main__':
    main()
