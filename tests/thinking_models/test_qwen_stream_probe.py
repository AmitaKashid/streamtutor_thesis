import base64
import json
import unittest
from unittest.mock import patch

from scripts.probe_qwen_stream import measure, percentile


class ProbeTests(unittest.TestCase):
    def test_measures_pcm_duration_and_waits_for_done(self):
        class Socket:
            def __init__(self):
                self.closed = False
                self.messages = iter([
                    json.dumps({'type':'audio_start'}),
                    json.dumps({'type':'audio_chunk','sample_rate':24000,
                                'pcm16_base64':base64.b64encode(bytes(48000)).decode()}),
                    json.dumps({'type':'audio_done'}),
                ])
            def send(self, payload): self.request = json.loads(payload)
            def recv(self): return next(self.messages)
            def close(self): self.closed = True
        socket = Socket()
        with patch('scripts.probe_qwen_stream.websocket.create_connection', return_value=socket), \
             patch('scripts.probe_qwen_stream.time.perf_counter', side_effect=[0., .25, .5]):
            result = measure('ws://test', 'Hello.', 'Aiden', 'en', 4, 10)
        self.assertEqual((result['ttfa_s'], result['audio_s'], result['audio_per_wall']), (.25, 1., 2.))
        self.assertEqual(socket.request['text'], 'Hello.')
        self.assertTrue(socket.closed)

    def test_percentile_interpolates_small_sample(self):
        self.assertEqual(percentile([1, 3, 2], .5), 2)
        self.assertAlmostEqual(percentile([1, 3], .95), 2.9)


if __name__ == '__main__': unittest.main()
