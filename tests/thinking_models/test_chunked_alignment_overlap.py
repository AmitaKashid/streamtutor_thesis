"""Chunked TTS must prepare the next segment while alignment is in progress."""
import threading
import unittest

import numpy as np
from test_speech_text_segments import run_pipeline


class ChunkedAlignmentOverlapTests(unittest.TestCase):
    def test_next_segment_synthesizes_while_previous_alignment_is_busy(self):
        alignment_started, second_started, release = (threading.Event() for _ in range(3))
        events, errors = [], []

        class TTS:
            def synthesize(self, **kwargs):
                if kwargs['text'].startswith('Second'):
                    second_started.set()
                return np.zeros((2400, 1), dtype=np.float32), 24000

        def align(text, audio, sample_rate):
            if text.startswith('First'):
                alignment_started.set()
                if not release.wait(3):
                    raise RuntimeError('test alignment gate timed out')
            return [{'text': text + ' ', 'start': 0., 'end': .05}]

        def run():
            try:
                run_pipeline('stream-chunked', TTS(), pieces=['First.', 'Second.'],
                             word_aligner=align, callback=lambda e, _: events.append(e))
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            self.assertTrue(alignment_started.wait(2))
            self.assertTrue(second_started.wait(2), 'next synthesis waited for alignment')
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual([e['text'] for e in events if e['type'] == 'audio_chunk'], ['First. ', 'Second. '])
        self.assertIn('audio_done', [e['type'] for e in events])

    def test_alignment_failure_does_not_report_success(self):
        class TTS:
            def synthesize(self, **kwargs):
                return np.zeros((2400, 1), dtype=np.float32), 24000
        def align(*_args):
            raise RuntimeError('broken aligner')
        with self.assertRaisesRegex(RuntimeError, 'broken aligner') as raised:
            run_pipeline('stream-chunked', TTS(), pieces=['First.', 'Second.'], word_aligner=align)
        types_seen = [event['type'] for event in raised.exception.events]
        self.assertIn('audio_stop', types_seen)
        self.assertNotIn('audio_done', types_seen)

    def test_interruption_unblocks_the_bounded_alignment_queue(self):
        started, fourth, release = (threading.Event() for _ in range(3))
        holder, events, errors = {}, [], []
        class TTS:
            def synthesize(self, **kwargs):
                if kwargs['text'].startswith('Segment 3'):
                    fourth.set()
                return np.zeros((2400, 1), dtype=np.float32), 24000
        def align(text, *_args):
            started.set()
            if not release.wait(3):
                raise RuntimeError('test alignment gate timed out')
            return [{'text': text + ' ', 'start': 0., 'end': .05}]
        def observe(event, playback):
            events.append(event)
            holder['playback'] = playback
        def run():
            try:
                run_pipeline('stream-chunked', TTS(), pieces=[f'Segment {n}.' for n in range(20)],
                             word_aligner=align, callback=observe)
            except Exception as exc:
                errors.append(exc)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            self.assertTrue(started.wait(2))
            self.assertTrue(fourth.wait(2))
            holder['playback'].interrupted.set()
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertIn('audio_stop', [event['type'] for event in events])
        self.assertNotIn('audio_done', [event['type'] for event in events])


if __name__ == '__main__':
    unittest.main()
