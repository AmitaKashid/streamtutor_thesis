"""Exercise token measurement against a local fake llama-server API."""
import json
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from unittest.mock import patch
from scripts import measure_teaching_prompt_tokens as measure


class TokenMeasureTest(unittest.TestCase):
    def test_all_modes_both_versions_and_no_completion_request(self):
        observed = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def send_json(self, payload):
                body = json.dumps(payload).encode()
                self.send_response(200); self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
            def do_GET(self):
                observed.append((self.command, self.path))
                self.send_json({'data': [{'id': 'qwen-test.gguf'}]})
            def do_POST(self):
                observed.append((self.command, self.path))
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path == '/apply-template':
                    self.server.assertions.append((body['messages'][0]['role'], body['chat_template_kwargs']['enable_thinking']))
                    self.send_json({'prompt': body['messages'][1]['content']})
                elif self.path == '/tokenize':
                    self.send_json({'tokens': list(range(len(body['content'].split())))})
                else:
                    self.send_error(404)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler); server.assertions = []
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory) / 'tokens.json'
                argv = ['measure', '--url', f'http://127.0.0.1:{server.server_port}', '--output', str(out)]
                with patch('sys.argv', argv), redirect_stdout(StringIO()): self.assertEqual(measure.main(), 0)
                data = json.loads(out.read_text(encoding='utf-8'))
                self.assertEqual(len(data['rows']), 8)
                self.assertEqual(set(row['version'] for row in data['rows']), {'baseline-v0', 'pedagogy-v1.0.0'})
                self.assertEqual(len({(row['style'], row['version']) for row in data['rows']}), 8)
                self.assertTrue(all(row['full_prompt_tokens'] > 0 for row in data['rows']))
                self.assertEqual(server.assertions, [('system', False)] * 8)
                self.assertEqual({path for method,path in observed}, {'/v1/models', '/apply-template', '/tokenize'})
        finally:
            server.shutdown(); server.server_close(); thread.join()

if __name__ == '__main__': unittest.main()
