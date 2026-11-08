"""Opt-in telemetry. No model imports or network access.

Values are pseudonymous timing events, NOT audio or transcript contents.
LLMTUTOR_TASK6_EVENTS_PATH must be set to enable output.
"""
from __future__ import annotations
import json, os, threading, time
from pathlib import Path
_LOCK = threading.Lock()

def record_event(session_id: str, event: str, turn_id: str | None = None) -> None:
    target = os.environ.get('LLMTUTOR_TASK6_EVENTS_PATH', '').strip()
    if not target:
        return
    if event not in {'tutor_turn_started','audio_chunk','audio_done','audio_stop',
                     'browser_audio_finished','browser_audio_interrupted'}:
        return
    row = {'session_id': session_id, 'event': event, 'turn_id': turn_id,
           'wall_time': time.time(), 'mono_ns': time.perf_counter_ns()}
    path = Path(target)
    try:
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    except OSError:
        # Observability is best-effort. Never break the tutor on telemetry I/O.
        return
