"""Offline, fail-closed analysis of the frozen StreamTutor judge-v3 experiment.

Usage: python -m judge.analyze_judge_selection_v3 --input-zip day4_task2_frozen_inputs.zip --output-dir judge_outputs/day4_comparison
Requires pandas and numpy. Never modifies input evidence.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

SPLITS = {'calibration':132, 'selection':792, 'robustness':123, 'contrasts':28}
MODELS = ('glm4_32b_0414_q4','compassjudger_2_7b_q4','fare_20b_q4','prometheus_7b_v2_q4')
QUALITY = ('critical','borderline','excellent')

def load_runs(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if any(n.startswith('/') or '..' in Path(n).parts for n in names):
            raise ValueError('Unsafe archive paths')
        groups = collections.defaultdict(dict)
        for name in names:
            chunks = name.replace('\\','/').split('/')
            if len(chunks)==2 and chunks[1] in {'manifest.json','predictions.jsonl','pairwise_predictions.jsonl'}:
                groups[chunks[0]][chunks[1]] = name
        runs = {}
        for group, files in groups.items():
            manifest = json.loads(z.read(files['manifest.json']))
            split = manifest.get('split') or ('contrasts' if 'contrasts' in group else None)
            if split is None: split='contrasts' if 'contrasts' in manifest['run_id'] else None
            # Pairwise manifests omit split.
            if split not in SPLITS: raise ValueError(f'Unknown split: {group} {split}')
            key = (manifest['judge_configuration']['model_id'],split)
            if key in runs: raise ValueError(f'Duplicate model/split: {key}')
            filekey = 'pairwise_predictions.jsonl' if split=='contrasts' else 'predictions.jsonl'
            if filekey not in files: raise ValueError(f'Missing predictions for {group}')
            raw = z.read(files[filekey]).splitlines()
            records = [json.loads(line) for line in raw if line.strip()]
            runs[key] = manifest,records
    if set(runs) != {(m,s) for m in MODELS for s in SPLITS}:
        raise ValueError(f'Incomplete run matrix: missing={set((m,s) for m in MODELS for s in SPLITS)-set(runs)}')
    return runs

def audit(runs):
    fingerprints={}; ids_by_split=collections.defaultdict(dict); rows=[]; benchmarks=[]
    common_dataset=None; common_git=None; shared_configs=None
    plans={}
    for (model,split),(manifest,records) in sorted(runs.items()):
        if manifest.get('status')!='completed': raise ValueError(f'Incomplete manifest {model}/{split}')
        if manifest.get('planned_evaluation_count')!=SPLITS[split] or len(records)!=SPLITS[split]:
            raise ValueError(f'Record count mismatch for {model}/{split}')
        dataset=manifest['dataset']; git=manifest.get('git_commit')
        dkey=(dataset.get('checksums_sha256'),dataset.get('rubric_sha256'))
        if common_dataset is None:common_dataset=dkey
        if dkey!=common_dataset:raise ValueError('Dataset/rubric hash mismatch')
        if common_git is None:common_git=git
        if git!=common_git:raise ValueError('Git revision mismatch')
        cfg=dict(manifest['judge_configuration']);cfg.pop('base_url',None);cfg.pop('model_id',None)
        if shared_configs is None:shared_configs=cfg
        if cfg!=shared_configs:raise ValueError('Decode/hardware configuration mismatch')
        plan=manifest.get('task_plan_sha256')
        if split in plans and plans[split]!=plan:raise ValueError('Task-plan mismatch')
        plans[split]=plan
        seen=set(); valid=0; retries=0; latencies=[]; end_to_end=[]
        for r in records:
            eid=r['evaluation_id'];h=r['input_sha256']
            if eid in seen:raise ValueError(f'Duplicate evaluation: {model} {split} {eid}')
            seen.add(eid)
            if r['judge_configuration']['model_id']!=model or r['rubric_sha256']!=dataset['rubric_sha256']:
                raise ValueError('Record provenance mismatch')
            if r.get('dataset_id')!=dataset['id'] or r.get('dataset_version')!=dataset['version']:
                raise ValueError('Record dataset mismatch')
            fp=(split,eid)
            if fp in fingerprints and fingerprints[fp]!=h:raise ValueError(f'Input-hash mismatch {fp}')
            fingerprints[fp]=h
            j=r['judge']; status=j['status']; result=j.get('result') or {}
            if status not in ('valid','invalid','error'):raise ValueError(f'Unknown status {status}')
            score=result.get('score') if status=='valid' else None
            if split!='contrasts' and status=='valid' and (type(score)!=int or not 1<=score<=5):
                raise ValueError(f'Invalid scalar score {eid}')
            if split=='contrasts' and status=='valid' and result.get('choice') not in (1,2):
                raise ValueError('Invalid pairwise choice')
            attempts=j.get('attempts') or [];natt=j.get('attempt_count',len(attempts))
            if len(attempts)!=natt:raise ValueError('Attempt-count mismatch')
            wall=[a.get('metrics',{}).get('wall_seconds') for a in attempts]
            wall=[float(x) for x in wall if isinstance(x,(float,int))]
            latencies+=wall
            if wall:end_to_end.append(sum(wall))
            retries+=max(natt-1,0);valid+=(status=='valid')
            quality=next((q for q in QUALITY if str(r.get('response_id','')).endswith('-'+q)),None)
            cm=r.get('case_metadata') or {}
            row={'model':model,'split':split,'evaluation_id':eid,'question_id':r.get('question_id'),
                 'response_id':r.get('response_id'), 'dimension':r.get('dimension',r.get('target_dimension')),
                 'quality':quality,'status':status,'score':score,'input_sha256':h,
                 'conversation_id':cm.get('conversation_id'), 'question_family':cm.get('question_family'),
                 'context_dependency':cm.get('context_dependency'), 'retrieval_condition':cm.get('retrieval_condition'),
                 'attempt_count':natt,'attempt_wall_sum_s':sum(wall) if wall else np.nan,
                 'contrast_id':r.get('contrast_id'),'order':r.get('order'),
                 'choice_source': None}
            if split=='contrasts' and status=='valid':
                row['choice_source']=r['candidate_1_source'] if result['choice']==1 else r['candidate_2_source']
            rows.append(row)
        ids_by_split[split][model]=seen
        benchmarks.append({'model':model,'split':split,'planned':SPLITS[split],'recorded':len(records),
            'valid':valid,'invalid':len(records)-valid,'first_attempt_valid':sum(x['judge']['status']=='valid' and x['judge'].get('attempt_count')==1 for x in records),
            'retry_count':retries,'p50_attempt_s':float(np.median(latencies)) if latencies else np.nan,
            'p95_attempt_s':float(np.percentile(latencies,95)) if latencies else np.nan,
            'p50_total_attempts_s':float(np.median(end_to_end)) if end_to_end else np.nan})
    for split,values in ids_by_split.items():
        if any(ids!=next(iter(values.values())) for ids in values.values()):raise ValueError(f'ID coverage mismatch: {split}')
    return pd.DataFrame(rows),pd.DataFrame(benchmarks), {'dataset_hash':common_dataset[0],'rubric_hash':common_dataset[1],'git_commit':common_git,'task_plan_hashes':plans}

def triples(df,split):
    d=df[(df.split==split)&df.quality.notna()&df.score.notna()].copy()
    # A triple is within the same question AND rubric dimension, across answer variants.
    a=d.pivot_table(index=['model','question_id','dimension'],columns='quality',values='score',aggfunc='first')
    a=a.dropna(subset=list(QUALITY)).reset_index()
    a['strict_order']=(a.excellent>a.borderline)&(a.borderline>a.critical)
    a['weak_order']=(a.excellent>=a.borderline)&(a.borderline>=a.critical)
    a['extreme_inversion']=a.excellent<a.critical
    a['gap']=a.excellent-a.critical
    return a

def paired(df):
    out=[]
    for split in ('calibration','selection','robustness'):
        d=df[(df.split==split)&(df.status=='valid')]
        p=d.pivot(index='evaluation_id',columns='model',values='score')
        for i,m1 in enumerate(MODELS):
            for m2 in MODELS[i+1:]:
                a=p[[m1,m2]].dropna()
                x=a[m1].to_numpy();y=a[m2].to_numpy();n=len(x)
                out.append({'split':split,'model_a':m1,'model_b':m2,'paired_n':n,
                   'exact_agreement':float(np.mean(x==y)),'mean_absolute_difference':float(np.mean(np.abs(x-y))),
                   'signed_mean_a_minus_b':float(np.mean(x-y)),
                   'critical_disagreement_ge_2':int(np.sum(np.abs(x-y)>=2))})
    return pd.DataFrame(out)

def contrast_summary(df):
    res=[]
    d=df[df.split=='contrasts']
    for m in MODELS:
        x=d[d.model==m]; p=x.pivot(index='contrast_id',columns='order',values='choice_source')
        ok=p.dropna(subset=['ab','ba']);consistent=(ok.ab==ok.ba)
        res.append({'model':m,'valid_judgments':int((x.status=='valid').sum()),'first_attempt_valid':int(((x.status=='valid')&(x.attempt_count==1)).sum()),
                    'complete_pairs':len(ok),'consistent_complete_pairs':int(consistent.sum()),
                    'preferred_a_complete_pairs':int(((ok.ab=='a')&(ok.ba=='a')).sum()),
                    'preferred_b_complete_pairs':int(((ok.ab=='b')&(ok.ba=='b')).sum()),
                    'note':'All provisional source preferences are a; no independent human labels'})
    return pd.DataFrame(res)

def cluster_bootstrap(trip,seed=20261108,nboot=2000):
    # Resample whole questions within each model, keeping their rubric dimensions together.
    rng=np.random.default_rng(seed);results=[]
    for m in MODELS:
        t=trip[trip.model==m]
        ids=t.question_id.unique(); groups=[t[t.question_id==q] for q in ids]
        if not len(groups):continue
        vals=[]
        for _ in range(nboot):
            idx=rng.integers(0,len(groups),size=len(groups))
            b=pd.concat([groups[k] for k in idx],ignore_index=True)
            vals.append((float(b.gap.mean()),float(b.weak_order.mean()),float(b.strict_order.mean())))
        arr=np.array(vals)
        results.append({'model':m,'n_question_clusters':len(ids),'n_triples':len(t),
                        'gap_mean':float(t.gap.mean()),'gap_ci_low':float(np.quantile(arr[:,0],.025)),
                        'gap_ci_high':float(np.quantile(arr[:,0],.975)),
                        'weak_order_rate':float(t.weak_order.mean()),'weak_ci_low':float(np.quantile(arr[:,1],.025)),
                        'weak_ci_high':float(np.quantile(arr[:,1],.975)),
                        'strict_order_rate':float(t.strict_order.mean()),'strict_ci_low':float(np.quantile(arr[:,2],.025)),
                        'strict_ci_high':float(np.quantile(arr[:,2],.975))})
    return pd.DataFrame(results)

def paired_cluster_bootstrap(trip, seed=20261108, nboot=2000):
    """Paired bootstrap differences: resample common question IDs, never independent model samples."""
    rng=np.random.default_rng(seed+1)
    out=[]
    for i,m1 in enumerate(MODELS):
        for m2 in MODELS[i+1:]:
            left=trip[trip.model==m1].set_index(['question_id','dimension'])
            right=trip[trip.model==m2].set_index(['question_id','dimension'])
            keys=left.index.intersection(right.index)
            a=left.loc[keys].reset_index(); b=right.loc[keys].reset_index()
            aligned=pd.DataFrame({'question_id':a.question_id,'gap_difference':a.gap.to_numpy()-b.gap.to_numpy(),
              'weak_difference':a.weak_order.astype(float).to_numpy()-b.weak_order.astype(float).to_numpy()})
            qids=aligned.question_id.unique(); grp=[aligned[aligned.question_id==q] for q in qids]
            samples=[]
            for _ in range(nboot):
                ix=rng.integers(0,len(qids),len(qids))
                draw=pd.concat([grp[k] for k in ix],ignore_index=True)
                samples.append((draw.gap_difference.mean(),draw.weak_difference.mean()))
            sims=np.asarray(samples)
            out.append({'model_a':m1,'model_b':m2,'common_question_clusters':len(qids),
              'common_triples':len(aligned),'gap_difference_a_minus_b':float(aligned.gap_difference.mean()),
              'gap_difference_ci_low':float(np.quantile(sims[:,0],.025)),
              'gap_difference_ci_high':float(np.quantile(sims[:,0],.975)),
              'weak_order_difference_a_minus_b':float(aligned.weak_difference.mean()),
              'weak_order_difference_ci_low':float(np.quantile(sims[:,1],.025)),
              'weak_order_difference_ci_high':float(np.quantile(sims[:,1],.975))})
    return pd.DataFrame(out)

def dimension_summary(df):
    d=df[(df.split=='selection') & df.quality.notna() & df.score.notna()]
    p=d.groupby(['model','dimension','quality']).score.agg(['mean','count']).reset_index()
    return p

def critical_disagreements(df,limit=60):
    d=df[(df.split=='selection') & (df.status=='valid')]
    p=d.pivot(index='evaluation_id',columns='model',values='score')
    spread=p.max(axis=1)-p.min(axis=1)
    salient=df[(df.split=='selection') & (df.dimension.isin(['correctness','groundedness','conversational_context']))]
    ids=set(salient.evaluation_id)
    res=p.loc[p.index.isin(ids)].copy();res['spread']=spread.loc[res.index]
    return res.sort_values('spread',ascending=False).head(limit).reset_index()

def run(input_zip,output_dir,nboot=2000):
    runs=load_runs(input_zip)
    df,metrics,provenance=audit(runs)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    t=triples(df,'selection'); r=triples(df,'robustness')
    reports={'aligned_predictions.csv':df,'run_metrics.csv':metrics,'selection_triples.csv':t,
        'selection_bootstrap.csv':cluster_bootstrap(t,nboot=nboot), 'selection_paired_bootstrap.csv':paired_cluster_bootstrap(t,nboot=nboot), 'robustness_triples.csv':r,
        'paired_agreement.csv':paired(df), 'dimension_summary.csv':dimension_summary(df),
        'contrast_summary.csv':contrast_summary(df),'salient_disagreements.csv':critical_disagreements(df)}
    for name,table in reports.items():table.to_csv(out/name,index=False)
    manifest={'source_zip_sha256':hashlib.sha256(Path(input_zip).read_bytes()).hexdigest(),
              'record_count':len(df),'run_count':len(runs),'seed':20261108,'bootstrap_repetitions':nboot,
              'provenance':provenance,'generated_files':sorted(reports)}
    (out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    return reports,manifest

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-zip',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--bootstrap-repetitions',type=int,default=2000)
    a=parser.parse_args();reports,manifest=run(a.input_zip,a.output_dir,a.bootstrap_repetitions)
    print(f"Audited {manifest['run_count']} runs and {manifest['record_count']} records; wrote {len(reports)} tables to {a.output_dir}")
if __name__=='__main__':main()
