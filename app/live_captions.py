"""Best-effort acoustic captions on CPU, independent of audio delivery.

Only the latest PCM snapshot per segment is queued. Slow/failed inference
cannot backpressure synthesis, and the browser owns the end-of-segment
fallback. There is no guessed word-rate clock and no Whisper GPU contention.
"""
from collections import OrderedDict
import logging
import threading

import numpy as np

log = logging.getLogger(__name__)
_model = None
_loading = False
_load_lock = threading.Lock()


def warm_live_captions():
    """Load only installed local assets; never download on a voice request."""
    global _loading
    with _load_lock:
        if _loading or _model is not None:
            return
        _loading = True

    def load():
        global _model
        try:
            from app.streaming_ctc import CpuAcousticModel
            _model = CpuAcousticModel()
            print('[CAPTIONS] CPU acoustic word tracker ready', flush=True)
        except Exception as exc:
            log.warning('CPU acoustic captions unavailable; using segment captions: %s', exc)
    threading.Thread(target=load, name='caption-model-load', daemon=True).start()


def live_caption_model(language):
    # This checkpoint is English-only. Never silently align another language.
    return _model if language in ('en', 'English') else None


class LiveCaptionWorker:
    def __init__(self, model, emit, cancelled):
        self.model, self.emit, self.cancelled = model, emit, cancelled
        self.pending = OrderedDict()
        self.condition = threading.Condition()
        self.closed = False
        self.finishing = False
        self.thread = threading.Thread(target=self._run, name='live-captions', daemon=True)
        self.thread.start()

    def start(self, text):
        # Tokenization (including a possible num2words import) belongs on the
        # worker as well: the synthesis thread must not wait for it.
        return {'text': text, 'tracker': None, 'disabled': False}

    def submit(self, segment_id, tracker, parts, sample_rate, final=False):
        # A shallow tuple snapshot is sufficient: PCM arrays are immutable
        # after receipt. Concatenation and inference happen on the worker.
        with self.condition:
            if self.closed or self.cancelled.is_set():
                return
            self.pending[segment_id] = (tracker, tuple(parts), sample_rate, final)
            while len(self.pending) > 2:
                self.pending.popitem(last=False)
            self.condition.notify()

    def close(self):
        # Do not join an inference call on the audio critical path.
        with self.condition:
            self.closed = True
            self.pending.clear()
            self.condition.notify()

    def finish(self):
        # Drain at most two coalesced snapshots in the background. The browser
        # may already have finished and will ignore those late updates.
        with self.condition:
            self.finishing = True
            self.condition.notify()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.finishing or bool(self.pending), timeout=.2)
                if self.closed or self.cancelled.is_set():
                    return
                if not self.pending:
                    if self.finishing:
                        return
                    continue
                segment_id, (tracker, parts, sr, final) = self.pending.popitem(last=False)
            try:
                if tracker['disabled']:
                    continue
                if tracker['tracker'] is None:
                    tracker['tracker'] = self.model.start(tracker['text'])
                if self.closed or self.cancelled.is_set():
                    return
                words = tracker['tracker'].update(
                    np.concatenate(parts, axis=0), sr, final=final,
                    cancelled=lambda: self.closed or self.cancelled.is_set())
                if not self.closed and not self.cancelled.is_set() and words:
                    self.emit({'type': 'audio_words', 'segment_id': segment_id, 'words': words})
            except Exception as exc:
                # A failed alignment affects captions only. Audio continues.
                tracker['disabled'] = True
                log.warning('Live caption alignment failed: %s', exc)
