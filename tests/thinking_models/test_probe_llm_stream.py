import ast
import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import probe_llm_stream as probe


class ProbeTests(unittest.TestCase):
    def test_default_probe_endpoints_match_web_model_routes(self):
        source = (Path(__file__).resolve().parents[2] / 'frontend_api.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'REMOTE_MODEL_URLS'
                                  for target in node.targets))
        routes = ast.literal_eval(assignment.value)
        self.assertEqual(set(probe.MODELS), set(routes) - {'llama3_8b_q5'})
        for model, (url, _protocol) in probe.MODELS.items():
            self.assertEqual(url, routes[model])

    def test_all_models_are_probed_and_thinking_is_enabled_only_where_supported(self):
        calls = []

        class Response:
            def raise_for_status(self): pass
            def json(self): return {'data': []}

        class Legacy:
            def __init__(self, url): self.url = url
            def stream(self, question):
                calls.append(('legacy', self.url, question))
                yield 'First '
                yield 'answer.'

        class Direct:
            def __init__(self, url): self.url = url
            def stream_with_cancel(self, question, cancel, *, thinking_enabled):
                calls.append(('direct', self.url, question, thinking_enabled))
                yield 'First '
                yield 'answer.'

        output = io.StringIO()
        with contextlib.redirect_stdout(output), \
             patch.object(probe, 'RemoteHTTPProvider', Legacy), \
             patch.object(probe, 'ThinkingHTTPProvider', Direct), \
             patch.object(probe.requests, 'get', return_value=Response()), \
             patch.object(probe, 'verify_model_identity', return_value='model.gguf'):
            rc = probe.main(['--model', 'all', '--thinking'])
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 5)
        self.assertEqual(sum(item[0] == 'legacy' for item in calls), 1)
        self.assertTrue(all(item[3] for item in calls if item[0] == 'direct'))
        self.assertEqual(output.getvalue().split('Summary:')[-1].count(': observed-multiple-deltas'), 5)

    def test_single_delta_is_inconclusive(self):
        class Legacy:
            def __init__(self, url): pass
            def stream(self, question): yield 'Entire response.'
        with contextlib.redirect_stdout(io.StringIO()), patch.object(probe, 'RemoteHTTPProvider', Legacy):
            self.assertEqual(probe.main(['--model', 'llama3_8b_q4']), 1)


if __name__ == '__main__':
    unittest.main()
