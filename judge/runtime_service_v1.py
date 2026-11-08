"""Crash-recoverable, bounded scalar-evaluation spool; independent of speech I/O.

All requests are immutable. A SQLite transaction admits work and enforces
idempotency; the worker calls an injected evaluator only after admission.
No model imports or network calls occur on the tutor thread.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Callable

from judge.runtime_contract_v1 import EvaluationRequest, EvaluationStatus, check_result, submission_decision


class ReferenceUnavailable(Exception):
    """No independently supplied reference; do not perform or retry scoring."""


class RuntimeJudgeService:
    def __init__(self, db_path: str | Path, evaluator: Callable[[EvaluationRequest], tuple[int, str]],
                 *, queue_limit: int = 32, max_attempts: int = 2, poll_interval: float = 0.10,
                 autostart: bool = True):
        self.db_path = str(db_path)
        self.evaluator = evaluator
        self.queue_limit = queue_limit
        self.max_attempts = max_attempts
        self.poll_interval = poll_interval
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript('''
              CREATE TABLE IF NOT EXISTS evaluations (
                id TEXT PRIMARY KEY, payload_hash TEXT NOT NULL, payload TEXT NOT NULL,
                status TEXT NOT NULL, reason TEXT, score INTEGER, feedback TEXT,
                attempts INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, updated_at REAL NOT NULL
              );
              CREATE TABLE IF NOT EXISTS attempts (
                id TEXT NOT NULL, number INTEGER NOT NULL, elapsed_s REAL NOT NULL,
                status TEXT NOT NULL, reason TEXT, PRIMARY KEY(id,number)
              );
            ''')
            # A previously running attempt may have been interrupted by a process crash.
            db.execute("UPDATE evaluations SET status='queued',reason='recovered_after_restart' WHERE status='running'")
        if autostart:
            self.start()

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.db_path, timeout=0.2)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def start(self):
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, daemon=True, name='runtime-scalar-judge')
        self._worker.start()

    def submit(self, request: EvaluationRequest) -> tuple[str, str]:
        """Return (evaluation-id, state); no judge inference, no blocking queue.put()."""
        key = request.idempotency_key
        payload = json.dumps(asdict(request), ensure_ascii=False, sort_keys=True)
        allowed, reason = submission_decision(request)
        now = time.time()
        with self._lock:
            try:
                with self._connect() as db:
                    existing = db.execute('SELECT payload_hash,status FROM evaluations WHERE id=?', (key,)).fetchone()
                    if existing:
                        if existing['payload_hash'] != request.payload_hash:
                            raise ValueError(f'payload conflict for evaluation {key}')
                        return key, existing['status']
                    backlog = db.execute("SELECT COUNT(*) FROM evaluations WHERE status IN ('queued','running')").fetchone()[0]
                    status = 'queued' if allowed and backlog < self.queue_limit else 'skipped'
                    if allowed and status == 'skipped':
                        reason = 'queue_full'
                    db.execute('INSERT INTO evaluations(id,payload_hash,payload,status,reason,created_at,updated_at) VALUES(?,?,?,?,?,?,?)',
                               (key,request.payload_hash,payload,status,None if status == 'queued' else reason,now,now))
                    return key, status
            except sqlite3.OperationalError as exc:
                # No work should be silently declared queued if persistence fails.
                return key, f'error:{type(exc).__name__}'

    def _run(self):
        while not self._stop.is_set():
            row = None
            with self._lock:
                try:
                    with self._connect() as db:
                        row = db.execute("SELECT id,payload,attempts FROM evaluations WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
                        if row:
                            db.execute("UPDATE evaluations SET status='running', updated_at=? WHERE id=?", (time.time(),row['id']))
                except sqlite3.OperationalError:
                    row = None
            if row is None:
                self._stop.wait(self.poll_interval)
                continue
            from judge.runtime_contract_v1 import EvaluationRequest, Evidence, TurnOutcome
            data = json.loads(row['payload'])
            data['outcome'] = TurnOutcome(data['outcome'])
            data['history'] = tuple(data['history'])
            data['evidence'] = tuple(Evidence(**e) for e in data['evidence'])
            request = EvaluationRequest(**data)
            started = time.perf_counter()
            try:
                score, feedback = self.evaluator(request)
                check_result(score, EvaluationStatus.VALID, feedback)
                status, reason = 'valid', None
            except ReferenceUnavailable as exc:
                score, feedback = None, None
                status, reason = 'skipped', f'reference_unavailable: {str(exc)[:240]}'
            except ValueError as exc:
                score, feedback = None, None
                status, reason = 'invalid', f'{type(exc).__name__}: {str(exc)[:240]}'
            except Exception as exc:
                score, feedback = None, None
                status, reason = 'error', f'{type(exc).__name__}: {str(exc)[:240]}'
            elapsed = time.perf_counter() - started
            attempt = row['attempts'] + 1
            retry = status not in ('valid', 'skipped') and attempt < self.max_attempts
            with self._lock:
                with self._connect() as db:
                    db.execute('INSERT OR REPLACE INTO attempts VALUES(?,?,?,?,?)', (row['id'],attempt,elapsed,status,reason))
                    db.execute('''UPDATE evaluations SET status=?,reason=?,score=?,feedback=?,attempts=?,updated_at=? WHERE id=?''',
                               ('queued' if retry else status,reason,score,feedback,attempt,time.time(),row['id']))

    def read(self, evaluation_id: str) -> dict | None:
        with self._connect() as db:
            row = db.execute('SELECT id,status,reason,score,feedback,attempts FROM evaluations WHERE id=?', (evaluation_id,)).fetchone()
            return dict(row) if row else None

    def close(self, *, timeout: float = 3.0):
        self._stop.set()
        if self._worker:
            self._worker.join(timeout=timeout)
