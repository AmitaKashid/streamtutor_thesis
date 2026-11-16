"""Do not silently accept a disconnected or indefinitely stalled TTS stream."""
import ast
import base64
import json
import types
import unittest
from pathlib import Path

import numpy as np


class SocketTimeout(Exception):
    pass


def make_engine(messages, clock):
    class Socket:
        def __init__(self): self.closed = False
        def settimeout(self, value): pass
        def send(self, value): pass
        def recv(self):
            item = messages.pop(0)
            if isinstance(item, Exception): raise item
            return item
        def close(self): self.closed = True
    socket = Socket()
    tree = ast.parse((Path(__file__).resolve().parents[2] / 'tts/qwen3_streaming_engine.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen3StreamingEngine')
    cls.bases = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
    env = {'websocket': types.SimpleNamespace(create_connection=lambda *a, **kw: socket),
           'WebSocketTimeoutException': SocketTimeout, 'time': types.SimpleNamespace(monotonic=lambda: next(clock)),
           'np': np, 'json': json, 'base64': base64}
    exec(compile(ast.fix_missing_locations(module), '<qwen-stream>', 'exec'), env)
    cfg = types.SimpleNamespace(stream_url='ws://test', speaker='Aiden', language='en', chunk_size=4, timeout_s=1)
    return env['Qwen3StreamingEngine'](cfg), socket, cfg


class QwenCompletionTests(unittest.TestCase):
    def test_stalled_stream_times_out_and_closes(self):
        engine, socket, cfg = make_engine([SocketTimeout()], iter([0., 1.1]))
        with self.assertRaisesRegex(TimeoutError, 'no data'):
            list(engine._stream_one_text('hello', cfg))
        self.assertTrue(socket.closed)

    def test_disconnect_after_audio_does_not_look_like_success(self):
        packet = json.dumps({'type': 'audio_chunk', 'sample_rate': 24000,
                             'pcm16_base64': base64.b64encode(b'\x00\x00').decode('ascii')})
        engine, socket, cfg = make_engine([packet, ''], iter([0., .1]))
        stream = engine._stream_one_text('hello', cfg)
        self.assertEqual(next(stream)[1], 24000)
        with self.assertRaisesRegex(RuntimeError, 'disconnected before completing'):
            next(stream)
        self.assertTrue(socket.closed)


if __name__ == '__main__':
    unittest.main()
