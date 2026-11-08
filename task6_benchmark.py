"""Task 6: paired runtime validation and latency audit. No third-party dependencies.

CMD (repository root):
  python task6_benchmark.py serve off --id off01
  python task6_benchmark.py serve on --id on01
  python task6_benchmark.py serve unavailable --id fail01
  python task6_benchmark.py report

Start one service mode at a time. Send the same scripted questions in the same
order, with a fresh browser conversation per measured turn. End each run via
Ctrl+C; output files are session-bounded by manifest start/end timestamps.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
import statistics
import sys
import time
from contextlib import closing

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'judge_outputs' / 'task6'
EVAL_DB = ROOT / 'judge_outputs' / 'runtime_v1_spool.sqlite'
RUN_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')

PROMPTS = [
    'What is entropy?',
    'What is entropy?',
    'What is entropy?',
    'What is entropy?',
    'What is entropy?',
]

# Optional scripted questions for exploratory coverage (not paired latency).
EXPLORATORY = [
    'Can you explain information gain using a small example?',
    'What does decision-tree pruning mean?',
    'Can you explain that with an example?',
]


def percentile(values, pct):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    position = (len(vals)-1)*pct/100
    lo = int(position)
    hi = min(lo+1,len(vals)-1)
    return round(vals[lo]+(position-lo)*(vals[hi]-vals[lo]),4)


def collect_logs(path):
    if not path.is_file():
        return []
    text = path.read_text(encoding='utf-8', errors='replace')
    sections = re.split(r'(?=^\[(?:TEXT|STT)\]\s)', text, flags=re.M)
    result = []
    patterns = {
      'stt_s': r'\[STT Full Transcription Latency\]\s*([0-9.]+)s',
      'rag_s': r'RAG latency\s*:\s*([0-9.]+)\s*s',
      'llm_ttft_s': r'LLM TTFT \(prompt -> first token\)\s*:\s*([0-9.]+)\s*s',
      'llm_completion_s': r'LLM full completion time\s*:\s*([0-9.]+)\s*s',
      'first_audio_pipeline_s': r'First audio from pipeline\s*:\s*([0-9.]+)\s*s',
      'tts_processing_s': r'TTS processing total\s*:\s*([0-9.]+)\s*s',
      'backend_total_s': r'End-to-end total\s*:\s*([0-9.]+)\s*s',
    }
    for section in sections:
        hit = re.match(r'^\[(?:TEXT|STT)\]\s*([^\n]*)',section)
        if not hit:
            continue
        record = {'question':hit.group(1).strip(), 'generation_completed':
                  bool(re.search(r'response status\s*:\s*completed', section))}
        for field, expr in patterns.items():
            match = re.search(expr,section)
            record[field] = float(match.group(1)) if match else None
        if any(record[f] is not None for f in patterns):
            result.append(record)
    return result


def collect_events(path):
    if not path.is_file():
        return [], []
    turns = defaultdict(list)
    invalid = 0
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            try:
                item = json.loads(line)
                if item.get('turn_id') and item.get('session_id'):
                    turns[(item['session_id'],item['turn_id'])].append(item)
            except (ValueError,TypeError,KeyError):
                invalid += 1
    rows = []
    for (session,turn), events in turns.items():
        start = next((e for e in events if e['event']=='tutor_turn_started'),None)
        if not start:
            continue
        def first(event):
            return next((e for e in events if e['event']==event),None)
        chunk = first('audio_chunk')
        done = first('audio_done')
        stop = first('audio_stop')
        finished = first('browser_audio_finished')
        interrupted = first('browser_audio_interrupted')
        terminal = finished or interrupted
        def delta(end):
            return round((end['mono_ns']-start['mono_ns'])/1e9,4) if end else None
        acks = [e for e in events if e['event'] in ('browser_audio_finished','browser_audio_interrupted')]
        rows.append({'session_id':session,'turn_id':turn,
            'first_chunk_s':delta(chunk),'backend_audio_done_s':delta(done),
            'browser_ack_s':delta(terminal),
            'browser_drain_after_done_s':
                round((finished['mono_ns']-done['mono_ns'])/1e9,4) if finished and done else None,
            'outcome':'delivered' if finished else 'interrupted' if interrupted or stop else 'missing_ack',
            'ack_count':len(acks),'conflicting_ack':bool(finished and interrupted),
            'ack_before_audio_done':bool(finished and done and finished['mono_ns'] < done['mono_ns']),
        })
    return rows, invalid


def collect_database(path, started, ended):
    if not path.is_file():
        return [], []
    with closing(sqlite3.connect(path)) as db:
        db.row_factory=sqlite3.Row
        evals=[dict(r) for r in db.execute(
            'SELECT id,status,reason,score,attempts,created_at,updated_at,payload '
            'FROM evaluations WHERE created_at >= ? AND created_at <= ? ORDER BY created_at',
            (started,ended))]
        ids = [x['id'] for x in evals]
        attempts=[]
        if ids:
            for pos in range(0,len(ids),400):
                batch=ids[pos:pos+400]
                sql='SELECT id,number,elapsed_s,status,reason FROM attempts WHERE id IN ('+','.join('?' for _ in batch)+')'
                attempts.extend(dict(r) for r in db.execute(sql,batch))
    for row in evals:
        data=json.loads(row.pop('payload'))
        for key in ('turn_id','evaluation_dimension','outcome','question_text','judge_model_id'):
            row[key]=data.get(key)
    return evals, attempts


def stats_row(manifest, turns, events, evals, attempts):
    counts=Counter(r['status'] for r in evals)
    eventcounts=Counter(e['outcome'] for e in events)
    row={'id':manifest['id'], 'condition':manifest['condition'],
         'questions_in_log':len(turns),'turns_with_events':len(events),
         'browser_delivered':eventcounts['delivered'],
         'browser_interrupted':eventcounts['interrupted'],
         'missing_ack':eventcounts['missing_ack'],
         'duplicate_ack_turns':sum(e['ack_count']>1 for e in events),
         'conflicting_ack_turns':sum(e['conflicting_ack'] for e in events),
         'premature_ack_turns':sum(e['ack_before_audio_done'] for e in events),
         'evaluation_rows':len(evals),
         'valid':counts['valid'],'skipped':counts['skipped'],
         'error':counts['error'],'invalid':counts['invalid'],
         'queued':counts['queued'],'running':counts['running'],
         'judge_attempts':len(attempts),
         'distinct_evaluated_turns':len({r['turn_id'] for r in evals}),
         'reference_unavailable':sum((r['reason'] or '').startswith('reference_unavailable') for r in evals),
         'queue_full':sum(r['reason']=='queue_full' for r in evals),
         'interrupt_skips':sum(r['reason']=='interrupted' for r in evals)}
    for metric,source in [
        ('backend_total', [r['backend_total_s'] for r in turns]),
        ('llm_ttft',[r['llm_ttft_s'] for r in turns]),
        ('first_audio_pipeline',[r['first_audio_pipeline_s'] for r in turns]),
        ('browser_completion',[e['browser_ack_s'] for e in events if e['outcome']=='delivered']),
        ('browser_queue_drain',[e['browser_drain_after_done_s'] for e in events if e['outcome']=='delivered']),
        ('judge_attempt',[r['elapsed_s'] for r in attempts]),
        ('judge_end_to_end',[r['updated_at']-r['created_at'] for r in evals if r['status']=='valid']),
    ]:
        samples=[x for x in source if x is not None]
        row[metric+'_n']=len(samples)
        row[metric+'_p50_s']=percentile(samples,50)
        row[metric+'_p95_s']=percentile(samples,95)
    return row


def report():
    manifests=sorted(OUT.glob('*.manifest.json'))
    if not manifests:
        raise SystemExit('No Task 6 sessions. Run serve off/on/unavailable first.')
    summaries=[]
    OUT.mkdir(parents=True, exist_ok=True)
    for path in manifests:
        m=json.loads(path.read_text(encoding='utf-8'))
        # A running session is bounded at report time; do not include later records.
        end=m.get('ended_at') or time.time()
        stem=m['id']
        logturns=collect_logs(OUT/(stem+'.log'))
        events, invalid=collect_events(OUT/(stem+'.events.jsonl'))
        evals,attempts=collect_database(EVAL_DB,m['started_at'],end)
        summary=stats_row(m,logturns,events,evals,attempts)
        summary['telemetry_parse_errors']=invalid
        summary['session_finalized']=bool(m.get('ended_at'))
        summaries.append(summary)
        for kind, rows in [('turns',logturns),('events',events),('evaluations',evals)]:
            with (OUT/(stem+'.'+kind+'.json')).open('w',encoding='utf-8') as fh:
                json.dump(rows,fh,indent=2)
    with (OUT/'summary.csv').open('w',encoding='utf-8',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(summaries[0]))
        writer.writeheader();writer.writerows(summaries)
    print('\nTASK 6 — MEASURED SUMMARY (do not conflate backend total and browser completion)')
    for row in summaries:
        print(f"{row['id']} [{row['condition']}] n_backend={row['backend_total_n']}, "
              f"backend p50={row['backend_total_p50_s']}s p95={row['backend_total_p95_s']}s; "
              f"browser n={row['browser_completion_n']} p50={row['browser_completion_p50_s']}s; "
              f"judge valid={row['valid']} skipped={row['skipped']} error={row['error']} invalid={row['invalid']}; "
              f"judge p50={row['judge_attempt_p50_s']}s; missing_ack={row['missing_ack']} "
              f"dup={row['duplicate_ack_turns']} premature={row['premature_ack_turns']}")
    print('Saved:',OUT/'summary.csv')
    print('NOTE: Start→browser ACK is server-received acknowledgment latency, not acoustic onset.')
    if any(r['backend_total_n']<5 or r['browser_completion_n']<5 for r in summaries if r['condition'] in ('off','on')):
        print('[INCOMPLETE] At least five non-interrupted fully played turns needed per OFF and ON condition.')
    on=[r for r in summaries if r['condition']=='on']
    off=[r for r in summaries if r['condition']=='off']
    if on and off and all(r['backend_total_n']>=5 and r['browser_completion_n']>=5 for r in on+off):
        a=[v['backend_total_p50_s'] for v in on if v['backend_total_p50_s'] is not None]
        b=[v['backend_total_p50_s'] for v in off if v['backend_total_p50_s'] is not None]
        if a and b:
            print('DESCRIPTIVE comparison of session medians (ON - OFF) backend seconds:',round(statistics.mean(a)-statistics.mean(b),4))
            print('Do NOT claim causal overhead without balanced repetitions and uncertainty.')


class Tee:
    def __init__(self,*streams): self.streams=streams
    def write(self,text):
        for stream in self.streams:stream.write(text);stream.flush()
    def flush(self):
        for stream in self.streams:stream.flush()
    def isatty(self): return False


def serve(condition,run_id):
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise SystemExit('Invalid session ID: alphanumeric, _ or - only')
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path=OUT/(run_id+'.manifest.json')
    if manifest_path.exists():
        raise SystemExit(f'Session ID exists: {manifest_path}. Use a new ID; data must never be overwritten.')
    # The task5 engineering fixture is isolated from frozen experiment evidence.
    try:
        from task5_acceptance import configure
    except ImportError as exc:
        raise SystemExit('task5_acceptance.py required in repository root') from exc
    configure()
    os.environ['LLMTUTOR_RUNTIME_JUDGE']='0'
    os.environ['LLMTUTOR_JUDGE_V1']='1' if condition!='off' else '0'
    if condition=='unavailable':
        os.environ['LLMTUTOR_JUDGE_URL']='http://127.0.0.1:18999'
    os.environ['LLMTUTOR_TASK6_EVENTS_PATH']=str(OUT/(run_id+'.events.jsonl'))
    if condition=='on':
        from judge.runtime_scalar_adapter_v3 import make_runtime_scalar_v3_adapter
        make_runtime_scalar_v3_adapter()
    manifest={'id':run_id,'condition':condition,'started_at':time.time(),
              'started_utc':datetime.now(timezone.utc).isoformat(),
              'judge_url':os.environ['LLMTUTOR_JUDGE_URL'],
              'rubric_sha256':os.environ['LLMTUTOR_JUDGE_RUBRIC_SHA256'],
              'prompt_sha256':os.environ['LLMTUTOR_JUDGE_PROMPT_SHA256'],
              'protocol':'task6-runtime-validation-v1',
              'prompts':PROMPTS,'notes':'engineering fixture; do not use as thesis gold evidence'}
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'\n[TASK 6] condition={condition} id={run_id}  log={OUT/(run_id+".log")}')
    print('[TASK 6] New browser conversation for EACH trial; microphone off for paired text-baseline')
    print('[TASK 6] Type exact question: "What is entropy?". Complete FIVE normal playback turns.')
    print('[TASK 6] Then stop with Ctrl+C, and run next condition. Do not stop GPU jobs.')
    import uvicorn
    original_out,original_err=sys.stdout,sys.stderr
    with (OUT/(run_id+'.log')).open('w',encoding='utf-8') as logfile:
        tee=Tee(original_out,logfile)
        sys.stdout=sys.stderr=tee
        try:
            uvicorn.run('frontend_api:app',host='127.0.0.1',port=8001,reload=False)
        finally:
            sys.stdout,sys.stderr=original_out,original_err
            manifest['ended_at']=time.time()
            manifest['ended_utc']=datetime.now(timezone.utc).isoformat()
            manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            print(f'[TASK 6] Finalized {run_id}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='cmd',required=True)
    s=sub.add_parser('serve')
    s.add_argument('condition',choices=('off','on','unavailable'))
    s.add_argument('--id',required=True)
    sub.add_parser('report')
    args=parser.parse_args()
    os.chdir(ROOT)
    serve(args.condition,args.id) if args.cmd=='serve' else report()

if __name__=='__main__': main()
