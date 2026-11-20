import threading
import time
import unittest
import numpy as np
from app.live_captions import LiveCaptionWorker
from app.streaming_ctc import prefix_path
from test_speech_text_segments import run_pipeline


class LiveCaptionTests(unittest.TestCase):
    def test_blocked_caption_inference_cannot_block_pcm_or_completion(self):
        started, release = threading.Event(), threading.Event()
        class Tracker:
            def update(self, *_args, **_kwargs):
                started.set()
                release.wait(3)
                return [{'text':'Hello. ', 'start':0, 'end':.1}]
        class Model:
            def start(self, _text): return Tracker()
        class TTS:
            def supports_streaming_input(self): return True
            def stream_text(self, *_args, **_kwargs):
                yield np.zeros((24000, 1), dtype=np.float32), 24000
                if not started.wait(2): raise RuntimeError('caption worker did not start')
                yield np.zeros((24000, 1), dtype=np.float32), 24000
        try:
            events = run_pipeline('stream-input', TTS(), pieces=['Hello.'], immediate=True,
                                  streaming_caption_model=Model())
            self.assertEqual(sum(e['type'] == 'audio_chunk' for e in events), 2)
            self.assertIn('audio_done', [e['type'] for e in events])
            self.assertNotIn('audio_words', [e['type'] for e in events])
        finally:
            release.set()

    def test_cancel_discards_pending_and_inflight_caption_results(self):
        started, release, cancel = (threading.Event() for _ in range(3))
        class Tracker:
            def update(self, *_args, **_kwargs):
                started.set(); release.wait(3)
                return [{'text':'late', 'start':0, 'end':.1}]
        events = []
        class Model:
            def start(self, _text): return Tracker()
        worker = LiveCaptionWorker(Model(), events.append, cancel)
        segment = worker.start('late')
        worker.submit('s', segment, [np.zeros(100)], 1000)
        self.assertTrue(started.wait(2))
        for n in range(20): worker.submit(str(n), worker.start('late'), [np.zeros(100)], 1000)
        self.assertLessEqual(len(worker.pending), 2)
        cancel.set(); worker.close(); release.set(); worker.thread.join(3)
        self.assertFalse(worker.thread.is_alive())
        self.assertEqual(events, [])

    def test_slow_tokenization_never_blocks_audio(self):
        started, release = threading.Event(), threading.Event()
        class Model:
            def start(self, _text):
                started.set()
                release.wait(3)
                raise ValueError('unsupported text')
        class TTS:
            def supports_streaming_input(self): return True
            def stream_text(self, *_args, **_kwargs):
                yield np.zeros((24000, 1), dtype=np.float32), 24000
                if not started.wait(2): raise RuntimeError('tokenizer did not start')
                yield np.zeros((24000, 1), dtype=np.float32), 24000
        try:
            events = run_pipeline('stream-input', TTS(), pieces=['Hello.'], immediate=True,
                                  streaming_caption_model=Model())
            self.assertEqual(sum(e['type'] == 'audio_chunk' for e in events), 2)
            self.assertIn('audio_done', [e['type'] for e in events])
        finally:
            release.set()

    def test_open_ctc_endpoint_does_not_force_an_unspoken_suffix(self):
        emissions = np.log(np.array([[.99,.005,.005],[.005,.99,.005],[.99,.005,.005]]))
        spans = prefix_path(emissions, np.array([1, 2]))
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0][:2], (1, 2))

    def test_repeated_ctc_characters_require_a_blank_and_silence_yields_no_text(self):
        silence = np.log(np.array([[.999,.001]] * 20))
        self.assertEqual(prefix_path(silence, np.array([1,1])), [])
        no_blank = np.log(np.array([[.001,.999]] * 2))
        self.assertEqual(len(prefix_path(no_blank, np.array([1,1]))), 1)
        with_blank = np.log(np.array([[.001,.999],[.999,.001],[.001,.999]]))
        self.assertEqual(len(prefix_path(with_blank, np.array([1,1]))), 2)


if __name__ == '__main__': unittest.main()
