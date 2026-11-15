"""Verify small HTTP events are delivered before the response body ends."""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llm.llmcli.providers.remote_http import RemoteHTTPProvider
from llm.llmcli.providers.thinking_http import ThinkingHTTPProvider


class HttpDeliveryTests(unittest.TestCase):
    def _exercise(self, first, rest, client, *, chunked=False):
        release = threading.Event()
        first_received = threading.Event()
        results, errors = [], []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get('Content-Length', '0')))
                if chunked:
                    self.protocol_version = 'HTTP/1.1'
                self.send_response(200)
                self.send_header('Content-Type', 'text/plain; charset=utf-8')
                # Non-chunked responses reproduce buffering by intermediaries.
                if chunked:
                    self.send_header('Transfer-Encoding', 'chunked')
                else:
                    self.send_header('Content-Length', str(len(first) + len(rest)))
                self.end_headers()
                def write(part):
                    if chunked:
                        self.wfile.write(f'{len(part):x}\r\n'.encode() + part + b'\r\n')
                    else:
                        self.wfile.write(part)
                    self.wfile.flush()
                # Deliberately split the multibyte emoji between network writes.
                emoji = first.find('😊'.encode('utf-8'))
                if emoji >= 0:
                    write(first[:emoji + 2])
                    write(first[emoji + 2:])
                else:
                    write(first)
                release.wait(3)
                try:
                    write(rest)
                    if chunked:
                        self.wfile.write(b'0\r\n\r\n')
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        server.daemon_threads = True
        serving = threading.Thread(target=server.serve_forever, daemon=True)
        serving.start()
        url = f'http://127.0.0.1:{server.server_address[1]}'

        def consume():
            try:
                for piece in client(url):
                    results.append(piece)
                    first_received.set()
            except Exception as exc:
                errors.append(exc)

        reader = threading.Thread(target=consume, daemon=True)
        reader.start()
        try:
            self.assertTrue(first_received.wait(1.5), 'first event was buffered until the stream ended')
            self.assertTrue(reader.is_alive(), 'the remainder should still be withheld')
        finally:
            release.set()
            reader.join(timeout=4)
            server.shutdown()
            server.server_close()
            serving.join(timeout=2)
        self.assertFalse(reader.is_alive())
        self.assertEqual(errors, [])
        return results

    def test_legacy_ndjson_yields_before_http_completion(self):
        first = (json.dumps({'delta': 'Hello 😊'}, ensure_ascii=False) + '\n').encode('utf-8')
        rest = (json.dumps({'delta': ' there'}) + '\n' +
                json.dumps({'done': True, 'finish_reason': 'stop'}) + '\n').encode()
        pieces = self._exercise(first, rest, lambda url: RemoteHTTPProvider(url).stream('question'))
        self.assertEqual(''.join(pieces), 'Hello 😊 there')

    def test_llama_server_sse_yields_before_http_completion(self):
        def event(delta, finish=None):
            return ('data: ' + json.dumps({'choices': [{'delta': delta, 'finish_reason': finish}]},
                                          ensure_ascii=False) + '\n\n').encode('utf-8')
        first = event({'content': 'Hello 😊'})
        rest = event({'content': ' there'}) + event({}, 'stop') + b'data: [DONE]\n\n'
        for chunked in (False, True):
            with self.subTest(chunked=chunked):
                pieces = self._exercise(first, rest, lambda url: ThinkingHTTPProvider(url).stream_with_cancel(
                    'question', None, thinking_enabled=False), chunked=chunked)
                self.assertEqual(''.join(pieces), 'Hello 😊 there')


if __name__ == '__main__':
    unittest.main()
