"""Model streaming must continue while a speech segment is synthesized."""
import threading
import unittest
import numpy as np

from test_speech_text_segments import run_pipeline


class LiveLlmAudioOverlapTests(unittest.TestCase):
    def test_interruption_unblocks_a_full_alignment_queue(self):
        alignment_started = threading.Event()
        release = threading.Event()
        fourth_synthesis = threading.Event()
        holder, events, errors = {}, [], []
        class TTS:
            def supports_streaming_input(self): return True
            def stream_text(self, text_events, **_kwargs):
                text = list(text_events)[0].text
                if text.startswith('Segment 3'):
                    fourth_synthesis.set()
                yield np.zeros((2400, 1), dtype=np.float32), 24000
        def align(text, audio, sample_rate):
            alignment_started.set()
            if not release.wait(3): raise RuntimeError('alignment did not release')
            return [{'text': text, 'start': 0., 'end': .05}]
        def observe(event, playback):
            events.append(event); holder['playback'] = playback
        def run():
            try:
                run_pipeline('stream-input', TTS(), pieces=[f'Segment {i}.' for i in range(20)],
                             word_aligner=align, callback=observe)
            except Exception as exc:
                errors.append(exc)
        thread = threading.Thread(target=run, daemon=True); thread.start()
        try:
            self.assertTrue(alignment_started.wait(2))
            self.assertTrue(fourth_synthesis.wait(2))
            holder['playback'].interrupted.set()
        finally:
            release.set(); thread.join(5)
        self.assertFalse(thread.is_alive()); self.assertEqual(errors, [])
        self.assertIn('audio_stop', [e['type'] for e in events])
        self.assertNotIn('audio_done', [e['type'] for e in events])

    def test_next_synthesis_can_overlap_prior_word_alignment(self):
        first_alignment = threading.Event()
        second_synthesis = threading.Event()
        release_alignment = threading.Event()
        events, errors = [], []

        class TTS:
            def supports_streaming_input(self): return True
            def stream_text(self, text_events, **_kwargs):
                text = list(text_events)[0].text
                if text.startswith('Second'):
                    second_synthesis.set()
                yield np.zeros((2400, 1), dtype=np.float32), 24000

        def align(text, audio, sample_rate):
            if text.startswith('First'):
                first_alignment.set()
                if not release_alignment.wait(3):
                    raise RuntimeError('alignment gate timed out')
            return [{'text': text, 'start': 0., 'end': .05}]

        def run():
            try:
                run_pipeline('stream-input', TTS(), pieces=['First.', 'Second.'],
                             word_aligner=align, callback=lambda event, _: events.append(event))
            except Exception as exc:
                errors.append(exc)
        thread = threading.Thread(target=run, daemon=True); thread.start()
        try:
            self.assertTrue(first_alignment.wait(2))
            self.assertTrue(second_synthesis.wait(2), 'next synthesis waited for alignment')
            self.assertFalse(any(event['type'] == 'audio_chunk' for event in events))
        finally:
            release_alignment.set(); thread.join(5)
        self.assertFalse(thread.is_alive()); self.assertEqual(errors, [])
        self.assertEqual([e['text'] for e in events if e['type'] == 'audio_chunk'], ['First. ', 'Second. '])
        self.assertIn('audio_done', [e['type'] for e in events])

    def test_raw_model_delta_is_emitted_before_speech_segment_is_ready(self):
        second_delta = threading.Event()
        release = threading.Event()
        events, errors = [], []

        class SentenceChunker:
            def __init__(self, _config):
                self.buffer = ''

            def feed(self, pieces):
                self.buffer += ''.join(pieces)
                if self.buffer.endswith('.'):
                    result, self.buffer = self.buffer, ''
                    yield result

            def flush(self):
                return self.buffer

        class Provider:
            def stream(self, _prompt):
                yield 'Hello'
                yield ' there'
                if not release.wait(3):
                    raise RuntimeError('model gate timed out')
                yield '.'

        class TTS:
            def supports_streaming_input(self): return True
            def stream_text(self, _events, **_kwargs):
                yield [1], 24000

        def observe(event, _playback):
            events.append(event)
            if event['type'] == 'llm_chunk' and event['text'] == ' there':
                second_delta.set()

        def run():
            try:
                run_pipeline('stream-input', TTS(), llm=Provider(), callback=observe,
                             chunker=SentenceChunker)
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            self.assertTrue(second_delta.wait(2), 'raw model delta was withheld for chunking')
            self.assertFalse(any(e['type'] == 'audio_chunk' for e in events))
        finally:
            release.set()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual([e['text'] for e in events if e['type'] == 'llm_chunk'],
                         ['Hello', ' there', '.'])
        self.assertEqual([e['text'] for e in events if e['type'] == 'audio_chunk'],
                         ['Hello there. '])

    def test_streaming_tts_does_not_block_model_deltas(self):
        tts_started = threading.Event()
        release_tts = threading.Event()
        second_delta = threading.Event()
        events, errors = [], []

        class Provider:
            def stream(self, _prompt):
                yield 'First segment. '
                if not tts_started.wait(2):
                    raise RuntimeError('TTS did not start')
                yield 'Second segment. '

        class SlowTTS:
            def supports_streaming_input(self):
                return True

            def stream_text(self, text_events, **_kwargs):
                text = list(text_events)[0].text
                if text.startswith('First'):
                    tts_started.set()
                    if not release_tts.wait(3):
                        raise RuntimeError('TTS gate timed out')
                yield [1], 24000

        def observe(event, _playback):
            events.append(event)
            if event['type'] == 'llm_chunk' and event['text'] == 'Second segment. ':
                second_delta.set()

        def run():
            try:
                run_pipeline('stream-input', SlowTTS(), llm=Provider(), callback=observe)
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            self.assertTrue(tts_started.wait(2), 'first audio segment never reached TTS')
            self.assertTrue(second_delta.wait(2), 'model reading stopped while TTS was busy')
            self.assertFalse(any(e['type'] == 'audio_chunk' for e in events),
                             'the TTS gate should still be holding the first audio')
        finally:
            release_tts.set()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual([e['text'] for e in events if e['type'] == 'llm_chunk'],
                         ['First segment. ', 'Second segment. '])
        self.assertEqual([e['text'] for e in events if e['type'] == 'audio_chunk'],
                         ['First segment. ', 'Second segment. '])
        self.assertIn('audio_done', [e['type'] for e in events])

    def test_llm_failure_cancels_inflight_synthesis_without_false_completion(self):
        tts_started = threading.Event()
        cancelled = threading.Event()
        events, errors = [], []

        class Provider:
            def stream(self, _prompt):
                yield 'First segment. '
                if not tts_started.wait(2):
                    raise RuntimeError('TTS did not start')
                raise RuntimeError('LLM failed')

        class CancellableTTS:
            def supports_streaming_input(self):
                return True

            def set_cancel_event(self, event):
                self.cancel_event = event

            def stream_text(self, _events, **_kwargs):
                tts_started.set()
                if self.cancel_event.wait(3):
                    cancelled.set()
                else:
                    raise RuntimeError('TTS did not receive cancellation')
                if False:
                    yield None

        def observe(event, _playback):
            events.append(event)

        def run():
            try:
                run_pipeline('stream-input', CancellableTTS(), llm=Provider(), callback=observe)
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertTrue(cancelled.is_set())
        self.assertEqual([str(error) for error in errors], ['LLM failed'])
        self.assertIn('audio_stop', [e['type'] for e in events])
        self.assertNotIn('audio_done', [e['type'] for e in events])

    def test_interruption_unblocks_a_full_speech_queue(self):
        tts_started = threading.Event()
        queue_filled = threading.Event()
        holder, events, errors = {}, [], []

        class Provider:
            def stream(self, _prompt):
                for number in range(200):
                    yield f'Segment {number}. '

        class WaitingTTS:
            def supports_streaming_input(self): return True
            def set_cancel_event(self, event): self.cancel = event
            def stream_text(self, _events, **_kwargs):
                tts_started.set()
                self.cancel.wait(3)
                if False:
                    yield None

        def observe(event, playback):
            events.append(event)
            holder['playback'] = playback
            if event['type'] == 'llm_chunk' and event['text'] == 'Segment 33. ':
                queue_filled.set()

        def run():
            try:
                run_pipeline('stream-input', WaitingTTS(), llm=Provider(), callback=observe)
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            self.assertTrue(tts_started.wait(2))
            self.assertTrue(queue_filled.wait(2), 'producer did not reach the bounded queue')
            holder['playback'].interrupted.set()
        finally:
            if holder:
                holder['playback'].interrupted.set()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertIn('audio_stop', [e['type'] for e in events])
        self.assertNotIn('audio_done', [e['type'] for e in events])


if __name__ == '__main__':
    unittest.main()
