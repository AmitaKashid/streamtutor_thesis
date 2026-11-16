"""Deterministic checks for preview scheduling, endpointing and first speech."""
import ast
import copy
import os
import threading
import time
import types
import unittest
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

from llm.llmcli.common.prosody_chunker import ProsodyChunker, ProsodyChunkerConfig
from stt.stt_types import TranscriptResult, Segment, Word

ROOT = Path(__file__).resolve().parents[2]


def load_method_nodes(path, names, namespace):
    """Exercise production nested functions without importing the GPU stack."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = []
    for name in names:
        fn = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == name)
        fn = copy.deepcopy(fn)
        fn.body = [ast.Global(names=node.names) if isinstance(node, ast.Nonlocal) else node
                   for node in fn.body]
        nodes.append(fn)
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)


class VoiceLatencyTests(unittest.TestCase):
    def test_whisper_alignment_and_stt_share_lock_but_http_stt_does_not(self):
        whisper_lock, http_lock, preview_lock = threading.Lock(), threading.Lock(), threading.Lock()
        preview = types.SimpleNamespace(engine='faster-whisper')
        env = {'_whisper_inference_lock': whisper_lock, 'stt_inference_lock': http_lock,
               '_preview_inference_lock': preview_lock, '_preview_backend': preview}
        load_method_nodes(ROOT / 'frontend_api.py', ['lock_for_stt'], env)
        self.assertIs(env['lock_for_stt'](types.SimpleNamespace(engine='faster-whisper')), whisper_lock)
        self.assertIs(env['lock_for_stt'](types.SimpleNamespace(engine='parakeet-http')), http_lock)
        self.assertIs(env['lock_for_stt'](preview), preview_lock)

    def test_preview_model_loads_in_background_and_falls_back_until_ready(self):
        loading = threading.Event(); release = threading.Event(); ready = threading.Event()
        calls = []
        class Model:
            def transcribe_partial(self, audio, sample_rate, language):
                calls.append(('warmup', len(audio), sample_rate, language))
                ready.set()
        model = Model()
        def create_backend(name, config):
            calls.append((name, config))
            loading.set()
            self.assertTrue(release.wait(3))
            return model
        env = {'os': os, 'threading': threading, 'np': np, 'TARGET_SR': 16000,
               'STT_LANGUAGE': 'en', 'create_backend': create_backend,
               '_preview_backend': None, '_preview_init_started': False,
               '_preview_init_lock': threading.Lock()}
        load_method_nodes(ROOT / 'frontend_api.py', ['start_preview_backend', 'select_preview_backend'], env)
        fallback = object()
        with patch.dict(os.environ, {'LLMTUTOR_PREVIEW_MODEL': 'tiny.en'}):
            env['start_preview_backend']()
            self.assertTrue(loading.wait(2))
            self.assertIs(env['select_preview_backend'](fallback), fallback)
            env['start_preview_backend']()  # no duplicate loads
            release.set()
            self.assertTrue(ready.wait(3))
            deadline = time.monotonic() + 2
            while env['select_preview_backend'](fallback) is not model and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertIs(env['select_preview_backend'](fallback), model)
        self.assertEqual(calls[0][0], 'faster-whisper')
        self.assertEqual(calls[0][1]['device'], 'cpu')
        self.assertEqual(calls[0][1]['model'], 'tiny.en')
        self.assertEqual(len([item for item in calls if item[0] == 'faster-whisper']), 1)

    def test_preview_uses_greedy_decoding_and_final_keeps_configured_beam(self):
        tree = ast.parse((ROOT / 'stt/stt_backends/faster_whisper_backend.py').read_text())
        cls = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FasterWhisperBackend'))
        cls.bases = []
        env = {'np': np, 'TranscriptResult': TranscriptResult, 'Segment': Segment, 'Word': Word}
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), '<whisper>', 'exec'), env)
        calls = []
        class Model:
            def transcribe(self, audio, **kwargs):
                calls.append(kwargs)
                return iter([types.SimpleNamespace(text='hello', start=0, end=1)]), types.SimpleNamespace(language='en')
        backend = env['FasterWhisperBackend'].__new__(env['FasterWhisperBackend'])
        backend.model = Model(); backend.beam_size = 5; backend.engine = 'faster-whisper'
        audio = np.zeros(16000, dtype=np.float32)
        self.assertEqual(backend.transcribe_partial(audio, 16000, 'en').text, 'hello')
        self.assertEqual(backend.transcribe(audio, 16000, 'en').text, 'hello')
        self.assertEqual([c['beam_size'] for c in calls], [1, 5])
        self.assertEqual([c['condition_on_previous_text'] for c in calls], [False, True])

    def test_short_preview_does_not_delay_first_valid_request(self):
        submitted = []
        env = {'os': os, 'partial_last_emit_t': 0., 'partial_frames': [b'\0' * 200],
               'partial_inflight': threading.Event(), 'partial_state_lock': threading.Lock(),
               'partial_generation': 0, 'final_stt_requested': threading.Event(),
               'PARTIAL_TRANSCRIBE_EVERY_S': .3, 'PARTIAL_MIN_AUDIO_S': .50, 'TARGET_SR': 16000,
               'time': types.SimpleNamespace(perf_counter=lambda: 10.0),
               'pcm16_bytes_to_float32': lambda raw: np.zeros(len(raw) // 2),
               'partial_executor': types.SimpleNamespace(submit=lambda *args: submitted.append(args)),
               'run_partial_transcription': lambda *args: None, 'stt_backend': object(),
               'select_preview_backend': lambda fallback: fallback}
        load_method_nodes(ROOT / 'frontend_api.py', ['maybe_emit_partial'], env)
        env['maybe_emit_partial']()
        self.assertEqual(env['partial_last_emit_t'], 0.)
        env['partial_frames'] = [b'\0' * 16000]
        env['maybe_emit_partial']()
        self.assertEqual(len(submitted), 1)
        self.assertEqual(env['partial_last_emit_t'], 10.)

    def test_endpoint_invalidates_running_preview_before_final_decode(self):
        started = threading.Event(); release = threading.Event(); emitted = []; finals = []
        model_lock = threading.Lock()
        class Engine:
            def transcribe_partial(self, **kwargs):
                started.set()
                self.test_assert_release = release.wait(3)
                return types.SimpleNamespace(text='old preview')
            def transcribe(self, **kwargs):
                finals.append('final')
                return types.SimpleNamespace(text='final text')
        engine = Engine()
        class Timer:
            def __init__(self): self.t0 = time.perf_counter()
            def elapsed(self): return time.perf_counter() - self.t0
        env = {'partial_last_text': '', 'partial_generation': 1,
               'partial_state_lock': threading.Lock(), 'partial_inflight': threading.Event(),
               'session_closed': threading.Event(), 'final_stt_requested': threading.Event(),
               'lock_for_stt': lambda e: model_lock, 'stt_backend': engine,
               'TARGET_SR': 16000, 'STT_LANGUAGE': 'en', 'time': time, 'os': os,
               'Timer': Timer, 'is_stt_hallucination_text': lambda text: False,
               'emit_from_thread': emitted.append}
        load_method_nodes(ROOT / 'frontend_api.py', ['run_partial_transcription', 'transcribe_final_timed'], env)
        audio = np.zeros(8000, dtype=np.float32)
        preview = threading.Thread(target=env['run_partial_transcription'], args=(audio, engine, 1))
        preview.start(); self.assertTrue(started.wait(2))
        result = []
        final = threading.Thread(target=lambda: result.append(env['transcribe_final_timed'](audio)))
        final.start(); self.assertTrue(env['final_stt_requested'].wait(2))
        release.set(); preview.join(3); final.join(3)
        self.assertFalse(preview.is_alive()); self.assertFalse(final.is_alive())
        self.assertEqual(emitted, [])
        self.assertEqual(finals, ['final'])
        self.assertEqual(result[0][0], 'final text')

    def test_browser_first_segment_can_end_at_short_sentence(self):
        cfg = ProsodyChunkerConfig(first_target_chars=70, later_target_chars=180,
                                   first_min_sentence_chars=28, min_sentence_chars=45)
        with patch('llm.llmcli.common.prosody_chunker.time.perf_counter', return_value=0):
            chunker = ProsodyChunker(cfg)
            chunks = list(chunker.feed(['A decision tree makes predictions. Next sentence.']))
        self.assertEqual(chunks, ['A decision tree makes predictions.'])
        self.assertEqual(chunker.flush(), 'Next sentence.')

    def test_second_segment_can_be_ready_before_a_long_later_segment(self):
        cfg = ProsodyChunkerConfig(first_min_sentence_chars=20, first_target_chars=70,
                                   second_target_chars=110, second_max_chars=150,
                                   later_target_chars=180, max_chars=220, max_wait_s=.3)
        with patch('llm.llmcli.common.prosody_chunker.time.perf_counter', return_value=10):
            chunker = ProsodyChunker(cfg)
            first = list(chunker.feed(['This is the first short sentence.']))
            self.assertEqual(first, ['This is the first short sentence.'])
            # With no punctuation and no timer advance, the special second max
            # is still lower than the ordinary later max.
            words = 'word ' * 34
            second = list(chunker.feed([words]))
            self.assertEqual(len(second), 1)
            self.assertLess(len(second[0]), 150)
            self.assertGreater(len(chunker.flush()), 0)

    def test_vad_reset_discards_active_speech_and_preroll(self):
        tree = ast.parse((ROOT / 'stt/vad_segmenter.py').read_text())
        nodes = [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.ClassDef)]
        class Vad:
            def __init__(self, aggressiveness): pass
            def is_speech(self, frame, sample_rate): return bool(frame[0])
        env = {'dataclass': dataclass, 'np': np, 'deque': deque, 'webrtcvad': types.SimpleNamespace(Vad=Vad),
               'TARGET_SR': 16000, 'FRAME_MS': 20, 'MAX_SILENCE_MS': 800, 'MAX_UTTERANCE_MS': 30000,
               'PRE_ROLL_MS': 250, '__name__': __name__}
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), '<vad>', 'exec'), env)
        vad = env['VADSegmenter']()
        vad.process_frame(bytes([0]) * 640, 0.)
        vad.process_frame(bytes([1]) * 640, .02)
        self.assertTrue(vad.in_speech)
        self.assertEqual(len(vad.buffer), 2)  # one pre-roll frame plus one onset frame
        vad.reset()
        self.assertFalse(vad.in_speech); self.assertEqual(vad.buffer, []); self.assertEqual(len(vad.pre), 0)
        self.assertIsNone(vad.flush())

    def test_browser_endpoint_waits_600ms_and_keeps_short_pauses(self):
        tree = ast.parse((ROOT / 'stt/vad_segmenter.py').read_text())
        nodes = [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.ClassDef)]
        class Vad:
            def __init__(self, aggressiveness): pass
            def is_speech(self, frame, sample_rate): return bool(frame[0])
        env = {'dataclass': dataclass, 'np': np, 'deque': deque, 'webrtcvad': types.SimpleNamespace(Vad=Vad),
               'TARGET_SR': 16000, 'FRAME_MS': 20, 'MAX_SILENCE_MS': 800, 'MAX_UTTERANCE_MS': 30000,
               'PRE_ROLL_MS': 250, '__name__': __name__}
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), '<vad>', 'exec'), env)
        vad = env['VADSegmenter'](silence_ms=600)
        voice, silence = bytes([1]) * 640, bytes(640)
        self.assertIsNone(vad.process_frame(voice, 0.0))
        for i in range(29):
            self.assertIsNone(vad.process_frame(silence, (i + 1) * .02))
        self.assertIsNone(vad.process_frame(voice, .60))  # an ordinary pause does not end the turn
        for i in range(29):
            self.assertIsNone(vad.process_frame(silence, .62 + i * .02))
        self.assertIsNotNone(vad.process_frame(silence, 1.20))


if __name__ == '__main__':
    unittest.main()
