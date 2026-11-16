"""Acoustic gating of normal and interruption transcripts, without model downloads."""
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
from stt.speech_evidence import SpeechEvidenceError, speech_audio
from test_voice_latency import ROOT, load_method_nodes


class SpeechEvidenceGuardTests(unittest.TestCase):
    def setUp(self):
        self.regions = []
        self.options = []
        vad = types.ModuleType('faster_whisper.vad')
        vad.VadOptions = lambda **kw: self.options.append(kw) or kw
        vad.get_speech_timestamps = lambda audio, options, sampling_rate: self.regions
        self.modules = patch.dict(sys.modules, {'faster_whisper': types.ModuleType('faster_whisper'),
                                               'faster_whisper.vad': vad})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_silence_and_non_speech_never_reach_a_recognizer(self):
        self.assertIsNone(speech_audio(np.zeros(16000), 16000))
        self.assertIsNone(speech_audio(np.full(16000, .1), 16000))
        self.assertEqual(self.options[-1]['min_speech_duration_ms'], 120)

    def test_speech_is_padded_but_pauses_are_not_cut(self):
        self.regions[:] = [{'start': 8000, 'end': 12000}, {'start': 18000, 'end': 24000}]
        audio = np.arange(32000, dtype=np.float32)
        np.testing.assert_array_equal(speech_audio(audio, 16000), audio[4800:27200])

    def test_detector_failure_is_not_silently_accepted(self):
        sys.modules['faster_whisper.vad'].get_speech_timestamps = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('VAD failed'))
        with self.assertRaises(SpeechEvidenceError):
            speech_audio(np.ones(16000), 16000)

    def test_preview_and_final_gate_before_whisper_or_any_other_backend(self):
        emitted, decoded = [], []
        class Engine:
            def transcribe(self, **kwargs):
                decoded.append(kwargs['audio'])
                return types.SimpleNamespace(text='Thank you.')
            transcribe_partial = transcribe
        class Timer:
            def __init__(self): self.t0 = time.perf_counter()
            def elapsed(self): return time.perf_counter() - self.t0
        env = {'partial_last_text': '', 'partial_generation': 1,
               'partial_state_lock': threading.Lock(), 'partial_inflight': threading.Event(),
               'session_closed': threading.Event(), 'final_stt_requested': threading.Event(),
               'lock_for_stt': lambda e: threading.Lock(), 'stt_backend': Engine(),
               'TARGET_SR': 16000, 'STT_LANGUAGE': 'en', 'time': time,
               'os': __import__('os'), 'Timer': Timer, 'speech_audio': speech_audio,
               'SpeechEvidenceError': SpeechEvidenceError, 'is_unusable_stt_text': lambda t: not t,
               'report_speech_evidence_error': lambda: emitted.append({'type': 'evidence_error'}),
               'emit_from_thread': emitted.append}
        load_method_nodes(ROOT / 'frontend_api.py', ['run_partial_transcription', 'transcribe_final_timed'], env)
        noise = np.full(16000, .1, dtype=np.float32)
        env['run_partial_transcription'](noise, env['stt_backend'], 1)
        self.assertEqual(env['transcribe_final_timed'](noise)[0], '')
        self.assertEqual(decoded, [])
        self.assertEqual(emitted, [])
        self.regions[:] = [{'start': 2000, 'end': 14000}]
        self.assertEqual(env['transcribe_final_timed'](noise)[0], 'Thank you.')
        self.assertEqual(len(decoded), 1)

    def test_detector_error_is_reported_without_stopping_the_tutor(self):
        events = []
        env = {'speech_evidence_error_reported': threading.Event(),
               'session_closed': threading.Event(), 'emit_from_thread': events.append}
        load_method_nodes(ROOT / 'frontend_api.py', ['report_speech_evidence_error'], env)
        env['report_speech_evidence_error']()
        env['report_speech_evidence_error']()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['type'], 'error')
        self.assertTrue(events[0]['nonfatal'])


if __name__ == '__main__':
    unittest.main()
