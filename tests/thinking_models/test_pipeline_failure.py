"""Exercise the real pipeline failure path without model/audio dependencies."""
import ast
import queue
import threading
import time
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock


def load_pipeline():
    source = (Path(__file__).resolve().parents[2] / 'app/mic_stt_llm_tts.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_llm_tts_pipeline')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), function], type_ignores=[])
    env = {
        'uuid': uuid, 'threading': threading, 'queue': queue, 'perf_counter': time.perf_counter,
        'HALLUCINATION_PHRASES': set(), 'print_effective_config': lambda args: None,
        'effective_config_payload': lambda args: {}, 'compose_prompt': lambda **kw: kw['user_query'],
        'start_terminal_dots': lambda *a: (threading.Event(), types.SimpleNamespace(join=lambda **kw: None)),
        'ProsodyChunkerConfig': lambda **kw: None,
        'ProsodyChunker': lambda cfg: types.SimpleNamespace(feed=lambda parts: parts, flush=lambda: ''),
        'StreamingTextChunk': lambda **kw: types.SimpleNamespace(**kw),
        'set_tts_state': lambda *a: None, 'Timer': lambda: types.SimpleNamespace(elapsed=lambda: 0.0),
    }
    exec(compile(ast.fix_missing_locations(module), '<pipeline-under-test>', 'exec'), env)
    return env['run_llm_tts_pipeline']


class PipelineFailureTest(unittest.TestCase):
    def _check(self, mode, tts_enabled):
        run = load_pipeline()
        playback = Mock()
        playback.interrupted = threading.Event()
        llm = types.SimpleNamespace(stream=lambda prompt: (_ for _ in ()).throw(RuntimeError('model failed')))
        class TTS:
            def supports_streaming_input(self): return True
            def stream_text(self, text, **kwargs):
                yield from text
        args = types.SimpleNamespace(min_words=1, tts_enabled=tts_enabled, mode=mode, teaching_style='default',
            chunk_first=80, chunk_mid=100, chunk_later=200, chunk_wait=1, browser_audio=True, voice_prompt=None)
        events = []
        active = threading.Event(); active.set()
        with self.assertRaisesRegex(RuntimeError, 'model failed'):
            run(transcript='test question', stt_time=0, total_timer=Mock(), llm=llm,
                tts_engine=TTS() if tts_enabled else None, retriever=None, args=args,
                playback=playback, pipeline_active=active, judge=None, event_cb=events.append)
        types_seen = [e['type'] for e in events]
        self.assertNotIn('audio_done', types_seen)
        self.assertNotIn('text_complete', types_seen)
        self.assertIn('audio_stop' if tts_enabled else 'text_interrupted', types_seen)
        self.assertFalse(active.is_set())

    def test_tts_worker_failure_does_not_complete_audio(self):
        run = load_pipeline()
        playback = Mock(); playback.interrupted = threading.Event()
        llm = types.SimpleNamespace(stream=lambda prompt: iter(['A complete sentence.']))
        class BrokenTTS:
            def synthesize(self, **kwargs): raise RuntimeError('Kokoro returned no audio')
        args = types.SimpleNamespace(min_words=1, tts_enabled=True, mode='stream', teaching_style='default',
            chunk_first=80, chunk_mid=100, chunk_later=200, chunk_wait=1, browser_audio=True, voice_prompt=None)
        events = []; active = threading.Event(); active.set()
        with self.assertRaisesRegex(RuntimeError, 'TTS synthesis failed'):
            run(transcript='test question', stt_time=0, total_timer=Mock(), llm=llm,
                tts_engine=BrokenTTS(), retriever=None, args=args, playback=playback,
                pipeline_active=active, judge=None, event_cb=events.append)
        types_seen = [event['type'] for event in events]
        self.assertIn('audio_stop', types_seen)
        self.assertNotIn('audio_done', types_seen)
        self.assertFalse(active.is_set())

    def test_emoji_only_chunk_is_not_sent_to_tts(self):
        run = load_pipeline()
        playback = Mock(); playback.interrupted = threading.Event()
        llm = types.SimpleNamespace(stream=lambda prompt: iter(['😊', 'A complete sentence.']))
        sent = []
        class TTS:
            def synthesize(self, **kwargs):
                sent.append(kwargs['text'])
                raise RuntimeError('synthetic failure')
        args = types.SimpleNamespace(min_words=1, tts_enabled=True, mode='stream', teaching_style='default',
            chunk_first=80, chunk_mid=100, chunk_later=200, chunk_wait=1, browser_audio=True, voice_prompt=None)
        events = []; active = threading.Event(); active.set()
        with self.assertRaisesRegex(RuntimeError, 'TTS synthesis failed'):
            run(transcript='test question', stt_time=0, total_timer=Mock(), llm=llm,
                tts_engine=TTS(), retriever=None, args=args, playback=playback,
                pipeline_active=active, judge=None, event_cb=events.append)
        self.assertEqual(sent, ['A complete sentence.'])
        self.assertTrue(any(e.get('type') == 'llm_chunk' and '😊' in e.get('text','') for e in events))

    def test_text_mode_failure_is_not_completed(self): self._check('stream', False)
    def test_stream_input_failure_is_not_completed(self): self._check('stream-input', True)
    def test_queued_tts_failure_is_not_completed(self): self._check('stream', True)

if __name__ == '__main__':
    unittest.main()
