"""Exercise packet delivery with synthesis and alignment deliberately blocked."""
import threading
import unittest

import numpy as np
from test_speech_text_segments import run_pipeline


class ImmediateAudioTests(unittest.TestCase):
    def test_first_pcm_reaches_browser_before_remaining_audio_exists(self):
        first_received, release = threading.Event(), threading.Event()
        events, errors = [], []

        class TTS:
            def supports_streaming_input(self): return True
            def stream_text(self, *_args, **_kwargs):
                yield np.zeros((2400, 1), dtype=np.float32), 24000
                if not release.wait(3):
                    raise RuntimeError('test gate timed out')
                yield np.ones((2400, 1), dtype=np.float32), 24000

        def observe(event, _playback):
            events.append(event)
            if event['type'] == 'audio_chunk': first_received.set()

        def forbidden_alignment(*_args):
            raise AssertionError('Whisper was invoked on the immediate audio path')

        def run():
            try:
                run_pipeline('stream-input', TTS(), immediate=True, pieces=['Hello.'],
                             word_aligner=forbidden_alignment, callback=observe)
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            self.assertTrue(first_received.wait(2), 'PCM waited for complete synthesis')
            self.assertNotIn('audio_done', [e['type'] for e in events])
        finally:
            release.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        packets = [e for e in events if e['type'] == 'audio_chunk']
        self.assertEqual(len(packets), 2)
        self.assertEqual([e['text'] for e in packets], ['Hello. ', None])
        self.assertEqual(packets[0]['text_sync'], 'segment')
        self.assertIn('audio_done', [e['type'] for e in events])

    def test_interruption_does_not_release_later_packets(self):
        class TTS:
            def supports_streaming_input(self): return True
            def stream_text(self, *_args, **_kwargs):
                for _ in range(10):
                    yield np.zeros((2400, 1), dtype=np.float32), 24000
        def interrupt(event, playback):
            if event['type'] == 'audio_chunk': playback.interrupted.set()
        events = run_pipeline('stream-input', TTS(), immediate=True,
                              pieces=['Hello.'], callback=interrupt)
        self.assertEqual(sum(e['type'] == 'audio_chunk' for e in events), 1)
        self.assertIn('audio_stop', [e['type'] for e in events])
        self.assertNotIn('audio_done', [e['type'] for e in events])


if __name__ == '__main__':
    unittest.main()
