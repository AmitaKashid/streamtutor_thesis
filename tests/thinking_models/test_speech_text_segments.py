"""Exercise the browser speech/text protocol through the real pipeline function."""
import ast
import contextlib
import io
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from app.word_alignment import WordAlignmentError

from test_pipeline_failure import load_pipeline


class FakeChunker:
    def __init__(self, _config):
        self.tail = ''

    def feed(self, pieces):
        for piece in pieces:
            if piece == 'tail':
                self.tail = 'Final tail.'
            else:
                yield piece

    def flush(self):
        return self.tail


class FakeTimer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def elapsed(self):
        return time.perf_counter() - self.t0


class StreamingTTS:
    def __init__(self, empty=False):
        self.requests = []
        self.empty = empty

    def supports_streaming_input(self):
        return True

    def stream_text(self, events, **_kwargs):
        events = list(events)
        self.requests.append([event.text for event in events])
        if not self.empty:
            yield [1], 24000
            yield [2], 24000


class ChunkedTTS:
    def __init__(self):
        self.requests = []

    def synthesize(self, **kwargs):
        self.requests.append(kwargs['text'])
        return [1], 24000


def run_pipeline(mode, tts, *, enabled=True, pieces=None, callback=None, llm=None, chunker=FakeChunker, word_aligner=None):
    run = load_pipeline()
    run.__globals__.update({
        'Timer': FakeTimer,
        'ProsodyChunker': chunker,
        'np_audio_to_pcm16_base64': lambda audio: ('pcm', 1),
        'torch_to_np_tc': lambda wav: wav,
        'np': np,
    })
    playback = Mock()
    playback.interrupted = threading.Event()
    if llm is None:
        llm = types.SimpleNamespace(stream=lambda prompt: iter(
            pieces if pieces is not None else ['First.', '😊', 'Again.', 'Again.', 'tail']
        ))
    args = types.SimpleNamespace(
        min_words=1, tts_enabled=enabled, mode=mode, teaching_style='default',
        chunk_first=80, chunk_mid=100, chunk_later=200, chunk_wait=1,
        browser_audio=True, voice_prompt=None,
    )
    events = []

    def emit(event):
        events.append(event)
        if callback:
            callback(event, playback)

    active = threading.Event()
    active.set()
    try:
        run(transcript='test question', stt_time=0, total_timer=FakeTimer(), llm=llm,
            tts_engine=tts, retriever=None, args=args, playback=playback,
            pipeline_active=active, judge=None, evaluation_turn_id='turn-1', event_cb=emit,
            word_aligner=word_aligner)
    except Exception as exc:
        exc.events = events
        raise
    return events


class SpeechTextSegmentsTest(unittest.TestCase):
    def test_browser_console_does_not_print_answer_before_speech(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            run_pipeline('stream-input', StreamingTTS(), pieces=['Do not reveal this.'])
        self.assertNotIn('Do not reveal this.', output.getvalue())

    def test_streaming_audio_is_aligned_before_word_timed_playback(self):
        seen = []
        def align(text, audio, sample_rate):
            seen.append((text, len(audio), sample_rate))
            return [{'text': text, 'start': 0.0, 'end': .1}]
        class AudioTTS:
            def supports_streaming_input(self): return True
            def stream_text(self, _events, **_kwargs):
                yield np.zeros((2400, 1), dtype=np.float32), 24000
                yield np.zeros((2400, 1), dtype=np.float32), 24000
        events = run_pipeline('stream-input', AudioTTS(), pieces=['Hello.'], word_aligner=align)
        packets = [e for e in events if e['type'] == 'audio_chunk']
        self.assertEqual(seen, [('Hello. ', 4800, 24000)])
        self.assertEqual(len(packets), 1)
        self.assertEqual(packets[0]['word_timings'], [{'text': 'Hello. ', 'start': 0.0, 'end': .1}])

    def test_chunked_audio_is_aligned_before_sending(self):
        class AudioTTS:
            def synthesize(self, **_kwargs): return np.zeros((4800, 1), dtype=np.float32), 24000
        events = run_pipeline('stream-chunked', AudioTTS(), pieces=['Hello.'],
                              word_aligner=lambda text, audio, sample_rate: [
                                  {'text': text + ' ', 'start': 0.0, 'end': .1}])
        packet = next(e for e in events if e['type'] == 'audio_chunk')
        self.assertEqual(packet['word_timings'][0]['text'], packet['text'])

    def test_interruption_during_alignment_does_not_send_audio(self):
        holder = {}
        def observe(event, playback):
            holder['playback'] = playback
        def align(text, audio, sample_rate):
            holder['playback'].interrupted.set()
            return [{'text': text, 'start': 0.0, 'end': 0.1}]
        events = run_pipeline('stream-input', StreamingTTS(), pieces=['Hello.'],
                              callback=observe, word_aligner=align)
        self.assertNotIn('audio_chunk', [event['type'] for event in events])
        self.assertIn('audio_stop', [event['type'] for event in events])
        self.assertNotIn('audio_done', [event['type'] for event in events])

    def test_alignment_failure_stops_speech_instead_of_showing_a_segment(self):
        def fail(*_args):
            raise RuntimeError('alignment failed')
        with self.assertRaisesRegex(RuntimeError, 'alignment failed') as raised:
            run_pipeline('stream-input', StreamingTTS(), pieces=['Hello.'], word_aligner=fail)
        event_types = [event['type'] for event in raised.exception.events]
        self.assertNotIn('audio_chunk', event_types)
        self.assertIn('audio_stop', event_types)
        self.assertNotIn('audio_done', event_types)

    def test_known_alignment_failure_preserves_audio_and_displays_text_afterwards(self):
        def fail(*_args):
            raise WordAlignmentError('cannot align this segment')
        events = run_pipeline('stream-input', StreamingTTS(), pieces=['Hello.'], word_aligner=fail)
        delivery = [event for event in events if event['type'] in {'audio_chunk', 'audio_text', 'alignment_warning'}]
        self.assertEqual([event['type'] for event in delivery], ['alignment_warning', 'audio_chunk', 'audio_text'])
        self.assertIsNone(delivery[1]['text'])
        self.assertEqual(delivery[2]['text'], 'Hello. ')
        self.assertIn('audio_done', [event['type'] for event in events])

    def test_alignment_error_on_formula_does_not_drop_later_speech(self):
        def align(text, audio, sr):
            if text.startswith('The entropy formula'):
                raise WordAlignmentError('token mismatch')
            return [{'text': text, 'start': 0., 'end': 0.}]
        events = run_pipeline('stream-input', StreamingTTS(),
                              pieces=['Entropy is uncertainty.', 'The entropy formula is minus p log p.', 'A fair coin has high entropy.'],
                              word_aligner=align)
        ordered = [e for e in events if e['type'] in {'audio_chunk', 'audio_text'}]
        self.assertEqual([e.get('text') for e in ordered],
                         ['Entropy is uncertainty. ', None, 'The entropy formula is minus p log p. ', 'A fair coin has high entropy. '])
        self.assertIn('audio_done', [e['type'] for e in events])

    def test_chunked_engine_also_falls_back_after_alignment_error(self):
        def fail(*_args):
            raise WordAlignmentError('timing unavailable')
        events = run_pipeline('stream-chunked', ChunkedTTS(), pieces=['Hello.'], word_aligner=fail)
        delivery = [e for e in events if e['type'] in {'audio_chunk', 'audio_text', 'alignment_warning'}]
        self.assertEqual([e['type'] for e in delivery], ['alignment_warning', 'audio_chunk', 'audio_text'])
        self.assertIsNone(delivery[1]['text'])
        self.assertEqual(delivery[2]['text'], 'Hello. ')
        self.assertIn('audio_done', [e['type'] for e in events])

    def test_web_config_routes_each_supported_engine_to_its_synthesis_api(self):
        source = (Path(__file__).resolve().parents[2] / 'frontend_api.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {'safe_int', 'build_pipeline_args'}]
        module = ast.Module(body=functions, type_ignores=[])
        env = {
            'SimpleNamespace': types.SimpleNamespace,
            'DEFAULT_LLM_MODEL': 'test', 'REMOTE_MODEL_URLS': {'test': 'http://test'},
            'RAG_TOP_K': 2, 'TEACHING_STYLE': 'socratic', 'TTS_BACKEND_NAME': 'kokoro',
            'STT_BACKEND_NAME': 'faster-whisper', 'ENABLE_RAG': False,
            'QWEN3_TTS_SPEAKER': 'Aiden', 'QWEN3_TTS_SPEAKERS': ['Aiden'],
            'QWEN3_TTS_LANGUAGE': 'en', 'QWEN3_TTS_LANGUAGES': ['en'],
        }
        exec(compile(ast.fix_missing_locations(module), '<web-config>', 'exec'), env)
        for backend, mode in [('kokoro', 'stream-chunked'), ('chatterbox', 'stream-chunked'),
                              ('chatterbox_stream', 'stream-input'), ('qwen3', 'stream-input')]:
            with self.subTest(backend=backend):
                self.assertEqual(env['build_pipeline_args']({'tts_backend': backend}).mode, mode)

    def test_streaming_tts_tags_only_first_audio_packet_for_each_segment(self):
        tts = StreamingTTS()
        events = run_pipeline('stream-input', tts)
        delivery = [e for e in events if e['type'] in ('audio_chunk', 'audio_text')]
        self.assertEqual([e.get('text') for e in delivery], [
            'First. ', None, '😊 ', 'Again. ', None, 'Again. ', None, 'Final tail. ', None,
        ])
        self.assertEqual(tts.requests, [['First. '], ['Again. '], ['Again. '], ['Final tail. ']])
        self.assertTrue(all(e['turn_id'] == 'turn-1' for e in delivery))
        types_seen = [e['type'] for e in events]
        self.assertGreater(types_seen.index('audio_done'), max(
            i for i, event in enumerate(events) if event['type'] in ('audio_chunk', 'audio_text')
        ))

    def test_chunked_tts_preserves_text_only_and_final_segments_in_audio_order(self):
        tts = ChunkedTTS()
        events = run_pipeline('stream-chunked', tts)
        delivery = [e for e in events if e['type'] in ('audio_chunk', 'audio_text')]
        self.assertEqual([e.get('text') for e in delivery],
                         ['First. ', '😊 ', 'Again. ', 'Again. ', 'Final tail. '])
        self.assertEqual(tts.requests, ['First.', 'Again.', 'Again.', 'Final tail.'])
        self.assertTrue(all(e['turn_id'] == 'turn-1' for e in delivery))

    def test_text_only_mode_keeps_immediate_turn_identified_llm_stream(self):
        events = run_pipeline('stream-chunked', None, enabled=False, pieces=['A ', 'short answer.'])
        chunks = [e for e in events if e['type'] == 'llm_chunk']
        self.assertEqual([e['text'] for e in chunks], ['A ', 'short answer.'])
        self.assertTrue(all(e['turn_id'] == 'turn-1' for e in chunks))
        self.assertNotIn('audio_chunk', [e['type'] for e in events])
        self.assertIn('text_complete', [e['type'] for e in events])

    def test_no_audio_fails_instead_of_falsely_completing_a_spoken_turn(self):
        with self.assertRaisesRegex(RuntimeError, 'Streaming TTS synthesis failed') as caught:
            run_pipeline('stream-input', StreamingTTS(empty=True), pieces=['Hello.'])
        self.assertIn('no audio', str(caught.exception.__cause__))
        self.assertIn('audio_stop', [e['type'] for e in caught.exception.events])
        self.assertNotIn('audio_done', [e['type'] for e in caught.exception.events])

    def test_empty_synthesized_buffer_does_not_complete_a_spoken_turn(self):
        class EmptyTTS:
            def synthesize(self, **_kwargs):
                return [], 24000

        with self.assertRaisesRegex(RuntimeError, 'TTS synthesis failed') as caught:
            run_pipeline('stream-chunked', EmptyTTS(), pieces=['Hello.'])
        self.assertIn('audio_stop', [e['type'] for e in caught.exception.events])
        self.assertNotIn('audio_done', [e['type'] for e in caught.exception.events])

    def test_interruption_stops_remaining_audio_and_segments(self):
        def stop_after_first(event, playback):
            if event['type'] == 'audio_chunk':
                playback.interrupted.set()
        events = run_pipeline('stream-input', StreamingTTS(), callback=stop_after_first)
        delivery = [e for e in events if e['type'] in ('audio_chunk', 'audio_text')]
        self.assertEqual([e['text'] for e in delivery], ['First. '])
        self.assertIn('audio_stop', [e['type'] for e in events])
        self.assertNotIn('audio_done', [e['type'] for e in events])


if __name__ == '__main__':
    unittest.main()
