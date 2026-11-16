"""A model length limit must not cancel already queued speech."""
import ast
import copy
import types
import unittest
from unittest.mock import patch

import numpy as np
from test_speech_text_segments import run_pipeline, StreamingTTS
from test_voice_latency import ROOT
from llm.llmcli.providers.remote_http import RemoteHTTPProvider
from test_legacy_native_metrics import Response


class GenerationLimitDeliveryTests(unittest.TestCase):
    def test_tutor_budget_is_larger_without_changing_judges_or_overrides(self):
        tree = ast.parse((ROOT / 'app/remote_llm_server.py').read_text(encoding='utf-8'))
        node = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_llm'))
        env = {'get_model_spec': lambda key: types.SimpleNamespace(gguf_path='model', chat_format=None,
               chat_template_kwargs=None, response_protocol=None, system_prompt=None),
               'LlamaCppProvider': lambda **kwargs: kwargs}
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), '<server-config>', 'exec'), env)
        tutor = env['build_llm']('llama3_8b_q4')
        self.assertEqual((tutor['n_ctx'], tutor['max_new_tokens']), (4096, 1024))
        judge = env['build_llm']('fare_20b_q4')
        self.assertEqual(judge['max_new_tokens'], 360)
        override = env['build_llm']('llama3_8b_q4', n_ctx_override=8192, max_new_tokens_override=500)
        self.assertEqual((override['n_ctx'], override['max_new_tokens']), (8192, 500))

    def test_truncated_model_drains_audio_and_warns_without_success_metrics(self):
        lines = [b'{"delta":"The first sentence."}',
                 b'{"delta":" A partial second sentence"}',
                 b'{"done":true,"finish_reason":"length","metrics":{"legacyBatchedEvalSeconds":1.0}}']
        spoken = []
        class TTS:
            def synthesize(self, **kwargs):
                spoken.append(kwargs['text'])
                return np.ones((100, 1), dtype=np.float32), 24000
        with patch('llm.llmcli.providers.remote_http.requests.post', return_value=Response(lines)):
            events = run_pipeline('stream-chunked', TTS(), llm=RemoteHTTPProvider('http://test'))
        self.assertEqual(spoken, ['The first sentence.', 'A partial second sentence'])
        self.assertIn('generation_warning', [event['type'] for event in events])
        self.assertIn('audio_done', [event['type'] for event in events])
        self.assertNotIn('audio_stop', [event['type'] for event in events])
        self.assertFalse(any('legacyBatchedEvalSeconds' in e.get('data', {}) for e in events))

    def test_successful_model_does_not_warn(self):
        class TTS:
            def synthesize(self, **kwargs): return np.ones((100, 1), dtype=np.float32), 24000
        events = run_pipeline('stream-chunked', TTS(), llm=types.SimpleNamespace(stream=lambda p: iter(['A complete answer.'])))
        self.assertNotIn('generation_warning', [event['type'] for event in events])

    def test_stream_input_also_preserves_queued_speech_at_limit(self):
        lines = [b'{"delta":"Some useful context."}', b'{"done":true,"finish_reason":"length"}']
        tts = StreamingTTS()
        with patch('llm.llmcli.providers.remote_http.requests.post', return_value=Response(lines)):
            events = run_pipeline('stream-input', tts, llm=RemoteHTTPProvider('http://test'))
        self.assertIn('Some useful context. ', tts.requests[0])
        self.assertIn('audio_done', [e['type'] for e in events])
        self.assertNotIn('audio_stop', [e['type'] for e in events])


if __name__ == '__main__':
    unittest.main()
