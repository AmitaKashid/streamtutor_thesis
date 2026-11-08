#!/usr/bin/env python3
"""Reproduce Task 7 from frozen ZIP: strict provenance, event pairing and judge audit.
Standard-library only. Example: python task7_reproduce.py task7_evidence_20261108.zip --out task7_reproduced
"""
import argparse, collections, csv, hashlib, io, json, pathlib, random, sqlite3, statistics, tempfile, zipfile

BLOCKS=[('b01_off','off'),('b02_on','on'),('b03_on','on'),('b04_off','off'),('b05_on','on'),('b06_off','off'),('b07_off','off'),('b08_on','on')]
DIMS={'clarity','completeness','conversational_context','correctness','groundedness','learner_appropriateness','pedagogy'}
FIELDS=['trial_id','question_id','condition','block_id','question','utc','outcome','session_id','turn_id','browser_ack_s','repetition','turn_to_first_audio_s','turn_to_audio_done_s','turn_to_browser_ack_s','audio_playback_after_generation_s','chunks','events_valid']
JFIELDS=['turn_id','dimension','status','score','attempts','created_at','updated_at','evaluator_wall_s','generated_chars','delivered_chars','delivery_completed','outcome','model']

def load_lines(z, suffix):
    names=[n for n in z.namelist() if n.replace('\\','/')=='task7/'+suffix]
    if len(names)!=1: raise ValueError(f'Expected exactly one {suffix}: {names}')
    return [json.loads(s) for s in z.read(names[0]).decode('utf-8-sig').splitlines() if s.strip()]

def load_name(z, suffix):
    names=[n for n in z.namelist() if n.replace('\\','/')=='task7/'+suffix or n==suffix]
    if len(names)!=1: raise ValueError(f'Expected exactly one {suffix}: {names}')
    return z.read(names[0])

def qtl(values,p):
    v=sorted(values); t=(len(v)-1)*p; lo=int(t); hi=min(lo+1,len(v)-1); return v[lo]+(v[hi]-v[lo])*(t-lo)

def report_rows(rows,metric):
    return {c:{'n':len(x),'mean':statistics.mean(x),'median':statistics.median(x),'p95':qtl(x,.95)} for c in ['off','on'] for x in [[r[metric] for r in rows if r['condition']==c]]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('archive',type=pathlib.Path);ap.add_argument('--out',type=pathlib.Path,default=pathlib.Path('task7_reproduced'));ap.add_argument('--expected-sha256',default='47426b73fad70fdc6635159d2e89d33b290219a8a72d975acf74581f8fc0b20f');a=ap.parse_args()
    actual=hashlib.sha256(a.archive.read_bytes()).hexdigest()
    if a.expected_sha256 and actual.lower()!=a.expected_sha256.lower(): raise ValueError(f'Archive SHA256 mismatch {actual}')
    with zipfile.ZipFile(a.archive) as z:
        if z.testzip() is not None: raise ValueError('Bad zip member')
        trials=[]; hashes=set(); registry=set()
        for block,condition in BLOCKS:
            manifest=json.loads(load_name(z,block+'.manifest.json'))
            if (manifest.get('block_id'),manifest.get('condition'))!=(block,condition): raise ValueError('Manifest mismatch '+block)
            hashes.add(manifest.get('schedule_sha256'));registry.add(manifest.get('registry_sha256'))
            results=load_lines(z,block+'.results.jsonl'); events=load_lines(z,block+'.events.jsonl')
            if len(results)!=11 or len({r['trial_id'] for r in results})!=11: raise ValueError('Trial count or duplicate '+block)
            expected=set(manifest['trial_ids'])
            if set(r['trial_id'] for r in results)!=expected: raise ValueError('Trial identity mismatch '+block)
            by_turn=collections.defaultdict(list)
            for e in events: by_turn[e['turn_id']].append(e)
            for r in results:
                if r['outcome']!='passed' or r['condition']!=condition or r['block_id']!=block: raise ValueError('Invalid trial '+str(r))
                ee=by_turn[r['turn_id']]
                if any(e['session_id']!=r['session_id'] for e in ee): raise ValueError('Session mismatch')
                kind=collections.defaultdict(list)
                for e in ee:kind[e['event']].append(e)
                if any(len(kind[t])!=1 for t in ('tutor_turn_started','audio_done','browser_audio_finished')) or len(kind['audio_chunk'])<1: raise ValueError('Missing/duplicate terminal event '+r['trial_id'])
                st=kind['tutor_turn_started'][0]['mono_ns']; first=kind['audio_chunk'][0]['mono_ns'];done=kind['audio_done'][0]['mono_ns'];ack=kind['browser_audio_finished'][0]['mono_ns']
                if not st<=first<=done<=ack: raise ValueError('Bad event order')
                metric=lambda x:(x-st)/1e9
                duration=(ack-st)/1e9
                if abs(duration-float(r['browser_ack_s']))>.002: raise ValueError('Browser duration mismatch '+r['trial_id'])
                rep=int(r['trial_id'].rsplit('_',1)[1]);trials.append(dict(r,repetition=rep,turn_to_first_audio_s=metric(first),turn_to_audio_done_s=metric(done),turn_to_browser_ack_s=duration,audio_playback_after_generation_s=(ack-done)/1e9,chunks=len(kind['audio_chunk']),events_valid=True))
        if len(hashes)!=1 or len(registry)!=1: raise ValueError('Inconsistent frozen hashes')
        if len(trials)!=88 or len({r['turn_id'] for r in trials})!=88: raise ValueError('Incorrect total trial counts')
        key=collections.Counter((r['question_id'],r['condition']) for r in trials)
        if len(key)!=22 or any(v!=4 for v in key.values()): raise ValueError('Unbalanced question-level workload')
        raw_db=load_name(z,'runtime_v1_spool.sqlite')
        with tempfile.TemporaryDirectory() as d:
            dbpath=pathlib.Path(d)/'spool.sqlite';dbpath.write_bytes(raw_db)
            con=sqlite3.connect(f'file:{dbpath}?mode=ro',uri=True)
            judge=[];on_turns={r['turn_id'] for r in trials if r['condition']=='on'}
            for payload,status,score,attempts,created,updated in con.execute('SELECT payload,status,score,attempts,created_at,updated_at FROM evaluations'):
                p=json.loads(payload)
                if p.get('turn_id') not in on_turns:continue
                judge.append({'turn_id':p['turn_id'],'dimension':p['evaluation_dimension'],'status':status,'score':score,'attempts':attempts,'created_at':created,'updated_at':updated,'evaluator_wall_s':updated-created,'generated_chars':len(p.get('generated_answer_text') or ''),'delivered_chars':len(p.get('delivered_answer_text') or ''),'delivery_completed':p.get('delivery_completed'),'outcome':p.get('outcome'),'model':p.get('judge_model_id')})
            con.close()
        if len(judge)!=308:raise ValueError(f'Unexpected FARE count {len(judge)}')
        for t in on_turns:
            j=[x for x in judge if x['turn_id']==t]
            if len(j)!=7 or {x['dimension'] for x in j}!=DIMS or any(x['status']!='valid' or x['attempts']!=1 or not isinstance(x['score'],int) or not 1<=x['score']<=5 for x in j):raise ValueError('FARE acceptance failed for '+t)
        trials.sort(key=lambda r:r['trial_id']);judge.sort(key=lambda r:(r['turn_id'],r['dimension']))
        a.out.mkdir(exist_ok=True,parents=True)
        for filename,fields,rr in [('task7_per_trial_analysis.csv',FIELDS,trials),('task7_judge_evaluations.csv',JFIELDS,judge)]:
            with (a.out/filename).open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rr)
        metrics={metric:report_rows(trials,metric) for metric in ['turn_to_first_audio_s','turn_to_audio_done_s','turn_to_browser_ack_s','audio_playback_after_generation_s','chunks']}
        paired={}
        for metric in metrics:
            qmeans={}
            for q in {r['question_id'] for r in trials}:
                qmeans[q]={c:statistics.mean(r[metric] for r in trials if r['question_id']==q and r['condition']==c) for c in ['on','off']}
            diffs=[vals['on']-vals['off'] for _,vals in sorted(qmeans.items())]
            rng=random.Random(20261108)
            boot=[statistics.mean(rng.choices(diffs,k=len(diffs))) for _ in range(20000)]
            paired[metric]={'question_count':len(diffs),'mean_question_difference_on_minus_off':statistics.mean(diffs),'question_bootstrap_95ci':[qtl(boot,.025),qtl(boot,.975)],'method':'question-cluster percentile bootstrap, 20,000 resamples, seed 20261108'}
        judge_by_turn=collections.defaultdict(list)
        for j in judge:judge_by_turn[j['turn_id']].append(j)
        post_ack=[max(j['updated_at'] for j in judge_by_turn[r['turn_id']])-max(e['wall_time'] for e in load_lines(z,r['block_id']+'.events.jsonl') if e['turn_id']==r['turn_id'] and e['event']=='browser_audio_finished') for r in trials if r['condition']=='on']
        summary={'archive_sha256':actual,'schedule_sha256':next(iter(hashes)),'registry_sha256':next(iter(registry)),'trial_count':len(trials),'on_trial_count':len(on_turns),'valid_judge_evaluations':len(judge),'metrics_by_condition':metrics,'paired_question_analysis':paired,'judge_all_seven_completed_post_playback_s':{'median':statistics.median(post_ack),'min':min(post_ack),'max':max(post_ack)}}
        (a.out/'task7_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print('PASS 88/88 trials; 308/308 valid FARE evaluations; SHA256',actual)
        for m in ['turn_to_first_audio_s','turn_to_browser_ack_s']:
            print(m,'OFF',round(metrics[m]['off']['median'],3),'ON',round(metrics[m]['on']['median'],3),'paired mean',round(paired[m]['mean_question_difference_on_minus_off'],4))
        print('Output:',a.out)
if __name__=='__main__':main()
