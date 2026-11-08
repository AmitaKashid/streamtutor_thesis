"""Task 7 single-block backend launcher. Execute from repository root.

Usage: python task7_serve.py --block b01_off
The browser runner is a separate process. Stop with Ctrl+C after its block.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'judge_outputs' / 'task7'

def load_block(block_id: str):
    import csv
    schedule = OUT / 'runtime_schedule.csv'
    with schedule.open(encoding='utf-8-sig', newline='') as f:
        rows = [row for row in csv.DictReader(f) if row['block_id'] == block_id]
    if not rows or len({r['condition'] for r in rows}) != 1:
        raise ValueError(f'Invalid or missing block {block_id}')
    if len({r['trial_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate trial IDs')
    return rows

def preflight(block: str, judge_url: str | None):
    rows = load_block(block)
    registry_path = (OUT / 'runtime_verified_references.json').resolve()
    provenance_path = (OUT / 'runtime_reference_provenance.json').resolve()
    from judge.runtime_scalar_adapter_v3 import exact_reference_registry
    references = exact_reference_registry(registry_path)
    provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
    for row in rows:
        question = row['question']
        if question not in references or question not in provenance:
            raise ValueError(f'Missing exact reference/provenance for {row["question_id"]}')
    if judge_url is not None and not judge_url.startswith('http'):
        raise ValueError('Judge URL must start http(s)')
    return rows, registry_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--block', required=True)
    parser.add_argument('--judge-url', default=None, help='FARE URL; defaults to Task 5 configured URL')
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--resume', action='store_true', help='Resume an existing block manifest without overwriting evidence')
    args = parser.parse_args()
    rows, registry = preflight(args.block, args.judge_url)
    condition = rows[0]['condition']
    manifest_path = OUT / f'{args.block}.manifest.json'
    old_manifest = None
    if manifest_path.exists():
        if not args.resume:
            raise SystemExit(f'Existing manifest; refusing overwrite: {manifest_path}. Use --resume for this exact block.')
        old_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        schedule_digest = hashlib.sha256((OUT/'runtime_schedule.csv').read_bytes()).hexdigest()
        registry_digest = hashlib.sha256(registry.read_bytes()).hexdigest()
        if (old_manifest.get('block_id') != args.block or
            old_manifest.get('condition') != condition or
            old_manifest.get('trial_ids') != [r['trial_id'] for r in rows] or
            old_manifest.get('schedule_sha256') != schedule_digest or
            old_manifest.get('registry_sha256') != registry_digest):
            raise SystemExit('REFUSING RESUME: manifest does not match frozen workload/registry')
    # IMPORTANT: ensure the port is free BEFORE preflight side effects or manifest writing.
    import socket
    probe = socket.socket()
    try:
        probe.settimeout(1)
        if probe.connect_ex(('127.0.0.1', args.port)) == 0:
            raise SystemExit(f'PORT {args.port} IS OCCUPIED: stop the old Task 7 server before starting {args.block}')
    finally:
        probe.close()
    from task5_acceptance import configure
    configure()
    # Task 5 configure() installs the engineering entropy fixture; replace it AFTER configure.
    os.environ['LLMTUTOR_REFERENCE_REGISTRY_JSON'] = str(registry)
    if args.judge_url:
        os.environ['LLMTUTOR_JUDGE_URL'] = args.judge_url
    os.environ['LLMTUTOR_RUNTIME_JUDGE'] = '0'
    os.environ['LLMTUTOR_JUDGE_V1'] = '1' if condition == 'on' else '0'
    events_path = (OUT / f'{args.block}.events.jsonl').resolve()
    if events_path.exists() and not args.resume:
        raise SystemExit(f'Existing telemetry; refusing overwrite: {events_path}. Use --resume for this exact block.')
    os.environ['LLMTUTOR_TASK6_EVENTS_PATH'] = str(events_path)
    if condition == 'on':
        from judge.runtime_scalar_adapter_v3 import make_runtime_scalar_v3_adapter
        adapter = make_runtime_scalar_v3_adapter()
        assert all(row['question'] in adapter.references for row in rows)
    data = dict(block_id=args.block, condition=condition, protocol='task7-runtime-v1',
        started_utc=datetime.now(timezone.utc).isoformat(),
        registry_sha256=hashlib.sha256(registry.read_bytes()).hexdigest(),
        schedule_sha256=hashlib.sha256((OUT/'runtime_schedule.csv').read_bytes()).hexdigest(),
        code_commit=os.popen('git rev-parse HEAD').read().strip(),
        judge_url=os.environ.get('LLMTUTOR_JUDGE_URL') if condition == 'on' else None,
        trial_ids=[r['trial_id'] for r in rows])
    if old_manifest is None:
        manifest_path.write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')
    else:
        print(f'RESUMING existing frozen manifest for {args.block}; prior evidence retained', flush=True)
    print(f'TASK7 {args.block}: {len(rows)} trials, condition={condition}; registry={registry}', flush=True)
    from fastapi import FastAPI
    import frontend_api
    app = frontend_api.app
    @app.get('/task7/block')
    def task7_block_identity():
        return {
            'protocol': 'task7-runtime-v1',
            'block_id': args.block,
            'condition': condition,
            'schedule_sha256': hashlib.sha256((OUT/'runtime_schedule.csv').read_bytes()).hexdigest(),
            'registry_sha256': hashlib.sha256(registry.read_bytes()).hexdigest(),
            'telemetry_path': str(events_path),
        }
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=args.port, reload=False)

if __name__ == '__main__':
    main()
