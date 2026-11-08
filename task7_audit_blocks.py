"""Audit Task 7 completed browser trials and async scalar judge scores.
Run from llmtutor repository root: python task7_audit_blocks.py b02_on
Only reads existing evidence. Does not change DB, manifests, or trials.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sqlite3
import sys

DIMENSIONS = frozenset(('correctness','clarity','completeness','pedagogy','learner_appropriateness','conversational_context','groundedness'))


def audit(block: str, root: Path) -> int:
    evidence=root/'judge_outputs'/'task7'
    schedule=evidence/'runtime_schedule.csv'
    results=evidence/f'{block}.results.jsonl'
    manifest=evidence/f'{block}.manifest.json'
    if not schedule.exists() or not results.exists() or not manifest.exists():
        print('FAIL missing schedule/results/manifest'); return 2
    with schedule.open(encoding='utf-8-sig', newline='') as stream:
        planned=[r for r in csv.DictReader(stream) if r['block_id']==block]
    if not planned:
        print('FAIL unrecognized block'); return 2
    condition=planned[0]['condition']
    m=json.loads(manifest.read_text(encoding='utf-8'))
    if m.get('condition')!=condition or m.get('block_id')!=block or m.get('trial_ids')!=[r['trial_id'] for r in planned]:
        print('FAIL schedule/manifest mismatch'); return 2
    records=[]
    for line in results.read_text(encoding='utf-8').splitlines():
        if line.strip():records.append(json.loads(line))
    by_id=defaultdict(list)
    for r in records:by_id[r.get('trial_id')].append(r)
    expected={r['trial_id'] for r in planned}
    unexpected=set(by_id)-expected
    missing=expected-set(by_id)
    invalid={tid:rs for tid,rs in by_id.items() if len(rs)!=1 or rs[0].get('outcome')!='passed' or not rs[0].get('turn_id') or not rs[0].get('session_id')}
    good=[r for r in records if r.get('outcome')=='passed' and r.get('turn_id') and r.get('session_id')]
    duplicate_turns=[turn for turn,n in Counter(r['turn_id'] for r in good).items() if n>1]
    print(f'BLOCK {block} condition={condition} planned={len(planned)} recorded={len(records)}')
    print(f'BROWSER passed={len(good)} missing={len(missing)} malformed_or_duplicate={len(invalid)} extra={len(unexpected)} duplicate_turns={len(duplicate_turns)}')
    if missing:print('MISSING',sorted(missing))
    if invalid:print('INVALID_TRIALS',{k:[r.get('outcome') for r in v] for k,v in invalid.items()})
    if unexpected:print('UNEXPECTED',sorted(unexpected))
    if duplicate_turns:print('DUPLICATE_TURNS',duplicate_turns)
    browser_ok=not(missing or invalid or unexpected or duplicate_turns) and len(good)==len(planned)
    if condition=='off':
        print('PASS browser-only OFF block' if browser_ok else 'FAIL OFF block browser integrity')
        return 0 if browser_ok else 1
    dbpath=root/'judge_outputs'/'runtime_v1_spool.sqlite'
    if not dbpath.exists():print('FAIL judge spool database missing'); return 2
    conn=sqlite3.connect(f'file:{dbpath.as_posix()}?mode=ro',uri=True)
    try:
        data=conn.execute('SELECT payload,status,score,reason FROM evaluations').fetchall()
    finally:conn.close()
    turn_ids={r['turn_id'] for r in good}
    judged=defaultdict(list)
    for payload,status,score,reason in data:
        p=json.loads(payload)
        if p.get('turn_id') in turn_ids:
            judged[p['turn_id']].append((p.get('evaluation_dimension'),status,score,reason,p.get('judge_model_id')))
    issues=[]
    aggregate=Counter()
    for r in good:
        items=judged.get(r['turn_id'],[])
        agg=Counter(d for d,*_ in items)
        aggregate.update(status for _,status,_,_,_ in items)
        valid=all(status=='valid' and isinstance(score,int) and 1<=score<=5 for _,status,score,_,_ in items)
        if len(items)!=7 or set(agg)!=DIMENSIONS or any(n!=1 for n in agg.values()) or not valid:
            issues.append((r['trial_id'],r['turn_id'],len(items),dict(Counter(x[1] for x in items)),sorted(set(DIMENSIONS)-set(agg)),[(d,s,reason) for d,s,_,reason,_ in items if s!='valid']))
    total=sum(len(v) for v in judged.values())
    print(f'FARE evaluations={total}/{len(planned)*7} statuses={dict(aggregate)} issues={len(issues)}')
    for issue in issues[:15]:print('ISSUE',issue)
    if browser_ok and not issues:
        print('PASS browser completions + 7 valid judge scores per ON trial');return 0
    print('FAIL/INCOMPLETE block acceptance — do not proceed until reviewed');return 1

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('block',nargs='+')
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    a=parser.parse_args()
    codes=[audit(b,a.root) for b in a.block]
    sys.exit(0 if all(c==0 for c in codes) else 1)
