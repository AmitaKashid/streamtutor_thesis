"""Idempotent, fail-closed, backup-first opt-in backend timing instrumentation.

Run: python task6_instrument.py install
     python task6_instrument.py check
Never changes browser protocol, scoring, or frozen rubric.
"""
from __future__ import annotations
import ast
from datetime import datetime
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
FILE = ROOT / 'frontend_api.py'
TAG = 'TASK6_TELEMETRY_V1'


def check():
    text = FILE.read_text(encoding='utf-8')
    if TAG not in text:
        raise SystemExit('Task 6 instrumentation not installed')
    ast.parse(text, filename=str(FILE))
    assert 'task6_record_event(evaluation_conversation_id, msg_type, payload.get("turn_id"))' in text
    assert 'task6_record_event(evaluation_conversation_id, action, turn_id)' in text
    print('[PASS] Task 6 telemetry hooks present and Python syntax valid')


def install():
    original = FILE.read_text(encoding='utf-8')
    if TAG in original:
        check()
        return
    anchor1 = '        msg_type = payload.get("type")\n'
    replacement1 = ('        msg_type = payload.get("type")\n'
                    '        # TASK6_TELEMETRY_V1: measurement only; never blocks speech on judge inference.\n'
                    '        task6_record_event(evaluation_conversation_id, msg_type, payload.get("turn_id"))\n')
    anchor2 = ('                resolve_browser_delivery(\n'
               '                    turn_id, delivered=action == "browser_audio_finished"\n'
               '                )\n')
    replacement2 = ('                task6_record_event(evaluation_conversation_id, action, turn_id)\n'
                    + anchor2)
    anchor3 = '    evaluation_conversation_id = f"web-{uuid.uuid4().hex}"\n'
    replacement3 = (anchor3 +
                    '    from task6_telemetry import record_event as task6_record_event\n')
    text = original
    for before, after in [(anchor1,replacement1),(anchor2,replacement2),(anchor3,replacement3)]:
        count = text.count(before)
        if count != 1:
            raise SystemExit(f'REFUSED: expected one unique anchor; found {count}: {before[:90]!r}. ' 
                             'Your frontend_api.py differs from the inspected Task 5 version.')
        text = text.replace(before,after,1)
    ast.parse(text, filename=str(FILE))
    backup_dir = ROOT / 'task6_backups'
    backup_dir.mkdir(exist_ok=True)
    backup = backup_dir / f'frontend_api.{datetime.now():%Y%m%d_%H%M%S}.py'
    shutil.copy2(FILE, backup)
    FILE.write_text(text, encoding='utf-8')
    print('[PASS] Opt-in telemetry installed in frontend_api.py; backup:', backup)
    check()

if __name__ == '__main__':
    if len(sys.argv)!=2 or sys.argv[1] not in {'install','check'}:
        raise SystemExit('Usage: python task6_instrument.py install|check')
    (install if sys.argv[1]=='install' else check)()
