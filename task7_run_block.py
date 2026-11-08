"""Task 7 REAL browser playback block runner.

Requires playwright: pip install playwright && python -m playwright install chromium
Launch task7_serve.py in another terminal first; Next.js frontend must be running.
Each trial uses a fresh browser context (zero prior turns). The pass criterion is
one server-side browser_audio_finished acknowledgement for the new turn.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
import re
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'judge_outputs' / 'task7'

def load_block(block):
    with (OUT/'runtime_schedule.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['block_id'] == block]
    if not rows or len({r['condition'] for r in rows}) != 1:
        raise ValueError(f'Unknown/invalid block: {block}')
    return rows

def events_for(path):
    if not path.exists(): return []
    result=[]
    with path.open(encoding='utf-8') as f:
        for line in f:
            try: result.append(json.loads(line))
            except json.JSONDecodeError: pass
    return result

def terminal(events, excluded):
    """Find a fresh start with a successful, unique browser ack, using event IDs."""
    starts = [e for e in events if e.get('event')=='tutor_turn_started'
              and e.get('turn_id') and (e.get('session_id'),e.get('turn_id')) not in excluded]
    for started in starts:
        key=(started['session_id'],started['turn_id'])
        related=[e for e in events if (e.get('session_id'),e.get('turn_id'))==key]
        done=[e for e in related if e.get('event')=='audio_done']
        ack=[e for e in related if e.get('event')=='browser_audio_finished']
        interrupted=[e for e in related if e.get('event') in ('audio_stop','browser_audio_interrupted')]
        if interrupted:
            return ('failed',key,'interrupted')
        if len(ack)>1:
            return ('failed',key,'duplicate ack')
        if len(ack)==1 and len(done)==1:
            if ack[0]['mono_ns'] < done[0]['mono_ns']:
                return ('failed',key,'ack before backend done')
            return ('passed',key,round((ack[0]['mono_ns']-started['mono_ns'])/1e9,4))
    return None

def validate_live_block(live, requested_block, requested_condition, manifest, expected_telemetry):
    """Reject stale or mismatched FastAPI server before any question is sent."""
    if (live.get('block_id') != requested_block or
        live.get('condition') != requested_condition or
        live.get('schedule_sha256') != manifest.get('schedule_sha256') or
        live.get('registry_sha256') != manifest.get('registry_sha256')):
        raise ValueError(
            f'WRONG LIVE BACKEND: requested {requested_block}/{requested_condition}, '
            f'but port 8001 serves {live.get("block_id")}/{live.get("condition")}. '
            'Stop previous task7_serve.py and start matching block.'
        )
    if str(Path(live.get('telemetry_path','')).resolve()) != str(Path(expected_telemetry).resolve()):
        raise ValueError('LIVE BACKEND TELEMETRY PATH MISMATCH: refuse to submit question')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--block',required=True)
    p.add_argument('--frontend',default='http://localhost:3000')
    p.add_argument('--timeout-s',type=int,default=180)
    p.add_argument('--headed', action='store_true', help='Use visible browser; recommended for real audio')
    p.add_argument('--max-trials', type=int, default=None, help='Stop after this many newly passed trials; use 1 for smoke test')
    p.add_argument('--retry-preflight', action='store_true', help='Allow retry of earlier failures that never produced a backend tutor_turn_started event')
    p.add_argument('--backend',default='http://127.0.0.1:8001',help='Backend HTTP origin for readiness check')
    args=p.parse_args()
    rows=load_block(args.block)
    # Diagnose backend liveness separately from frontend WebSocket UI status.
    try:
        with urllib.request.urlopen(args.backend.rstrip('/')+'/health', timeout=5) as response:
            health=json.loads(response.read().decode('utf-8'))
            if response.status!=200 or health.get('ok') is not True:
                raise RuntimeError(f'Unexpected backend health response: {health}')
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f'BACKEND NOT READY at {args.backend}/health: {type(exc).__name__}: {exc}') from exc
    manifest=OUT/f'{args.block}.manifest.json'
    if not manifest.exists():raise SystemExit('Start matching task7_serve.py block first')
    m=json.loads(manifest.read_text(encoding='utf-8'))
    if m['condition']!=rows[0]['condition'] or m['trial_ids']!=[r['trial_id'] for r in rows]:
        raise SystemExit('Server manifest does not match requested schedule block')
    # Fail closed unless the *live* FastAPI process identifies itself as this block.
    # A disk manifest alone cannot distinguish an old OFF backend from a new ON run.
    try:
        with urllib.request.urlopen(args.backend.rstrip('/')+'/task7/block', timeout=5) as response:
            live = json.loads(response.read().decode('utf-8'))
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError) as exc:
        raise SystemExit(f'BLOCK IDENTITY UNAVAILABLE on port 8001: {exc}. Stop stale backend and start guarded task7_serve.py.') from exc
    try:
        validate_live_block(live, args.block, rows[0]['condition'], m, OUT/f'{args.block}.events.jsonl')
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    telemetry=OUT/f'{args.block}.events.jsonl'
    # Telemetry is created lazily by task6_telemetry.record_event() on the first turn.
    # An absent file before the first question is normal; events_for() returns [].
    results=OUT/f'{args.block}.results.jsonl'
    if results.exists():
        completed=[json.loads(l) for l in results.read_text(encoding='utf-8').splitlines() if l.strip()]
    else: completed=[]
    finished_ids={r['trial_id'] for r in completed if r.get('outcome')=='passed'}
    failed=[r for r in completed if r.get('outcome')!='passed']
    if failed and not args.retry_preflight:
        raise SystemExit('Earlier failed trial. Use --retry-preflight ONLY for an unsent browser-readiness failure.')
    if failed:
        starts=[e for e in events_for(telemetry) if e.get('event')=='tutor_turn_started']
        if starts or any(r.get('session_id') or r.get('turn_id') for r in failed):
            raise SystemExit('Previous failure may have submitted a turn; do not auto-retry. Inspect telemetry first.')
        if any('Frontend did not show Backend: Online' not in r.get('error','') and 'Locator.wait_for' not in r.get('error','') for r in failed):
            raise SystemExit('Prior failure was not a known pre-submit readiness failure; manual investigation required.')
    # Already-observed turn IDs cannot be mistaken for this trial's playback.
    existing=events_for(telemetry)
    excluded={(e.get('session_id'),e.get('turn_id')) for e in existing if e.get('event')=='tutor_turn_started'}
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as ex:
        raise SystemExit('Install Playwright: python -m pip install playwright; python -m playwright install chromium') from ex
    with sync_playwright() as playwright:
        browser=playwright.chromium.launch(headless=not args.headed, args=['--autoplay-policy=no-user-gesture-required'])
        try:
            passed_in_this_run=0
            for row in rows:
                if row['trial_id'] in finished_ids:continue
                context=browser.new_context()
                page=context.new_page()
                ws_events=[]
                def record_ws(ws):
                    ws_events.append('websocket_open: '+ws.url)
                    ws.on('close', lambda: ws_events.append('websocket_closed: '+ws.url))
                    ws.on('socketerror', lambda err: ws_events.append('websocket_error: '+str(err)))
                page.on('websocket', record_ws)
                page_errors=[]
                page.on('pageerror',lambda err: page_errors.append(str(err)))
                outcome={'trial_id':row['trial_id'],'question_id':row['question_id'],
                         'condition':row['condition'],'block_id':args.block,
                         'question':row['question'],'utc':datetime.now(timezone.utc).isoformat()}
                try:
                    page.goto(args.frontend,wait_until='domcontentloaded',timeout=30000)
                    # React may render the status in a nested span; compare its
                    # textContent rather than depending on Playwright exact text.
                    # This remains a real frontend WebSocket readiness gate.
                    try:
                        page.wait_for_function(
                            r"""() => [...document.querySelectorAll('span')]
                                   .some(el => /^Backend:\s*Online$/i.test((el.textContent||'').trim()))""",
                            timeout=35000,
                        )
                    except Exception as readiness_error:
                        diag_dir=OUT/'diagnostics'
                        diag_dir.mkdir(exist_ok=True, parents=True)
                        png=diag_dir/f"{row['trial_id']}.png"
                        txt=diag_dir/f"{row['trial_id']}.txt"
                        try: page.screenshot(path=str(png), full_page=True)
                        except Exception: pass
                        body=page.locator('body').inner_text(timeout=5000)
                        txt.write_text(
                            f'page_url={page.url}\nbackend_health={args.backend}/health OK\n'
                            f'expected_websocket=ws://127.0.0.1:8001/ws\n'
                            f'websocket_events={ws_events!r}\n'
                            f'page_errors={page_errors!r}\n'
                            f'page_body_first_3000_chars=\n{body[:3000]}\n',
                            encoding='utf-8',
                        )
                        raise RuntimeError(
                            f'Frontend did not show Backend: Online within 35s. '
                            f'Backend HTTP is healthy; inspect {txt} and {png}. '
                            f'Probable frontend WebSocket connection problem or outdated UI build.'
                        ) from readiness_error
                    question_input=page.get_by_placeholder('Type a question, or use the microphone...')
                    question_input.fill(row['question'])
                    question_input.press('Enter')
                    deadline=time.monotonic()+args.timeout_s
                    result=None
                    while time.monotonic()<deadline:
                        result=terminal(events_for(telemetry),excluded)
                        if result is not None: break
                        time.sleep(.2)
                    if result is None:
                        outcome.update(outcome='failed',error='timeout waiting browser ACK')
                    else:
                        state,key,value=result
                        excluded.add(key)
                        outcome.update(outcome=state,session_id=key[0],turn_id=key[1])
                        if state=='passed':outcome['browser_ack_s']=value
                        else:outcome['error']=value
                except Exception as ex:
                    outcome.update(outcome='failed',error=f'{type(ex).__name__}: {ex}')
                finally:
                    context.close()
                    with results.open('a',encoding='utf-8') as f:
                        f.write(json.dumps(outcome,ensure_ascii=False)+'\n')
                print(f"{row['trial_id']}: {outcome['outcome']} {outcome.get('error','')}",flush=True)
                if outcome['outcome']!='passed':
                    raise SystemExit('BLOCK STOPPED: inspect error before any continuation')
                passed_in_this_run+=1
                if args.max_trials is not None and passed_in_this_run>=args.max_trials:
                    print(f'SMOKE COMPLETE: {passed_in_this_run} new successful trial(s); run again without --max-trials for remainder',flush=True)
                    return
        finally:
            browser.close()
    print(f'BLOCK {args.block} COMPLETE: {len(rows)} scheduled trials')

if __name__=='__main__': main()
