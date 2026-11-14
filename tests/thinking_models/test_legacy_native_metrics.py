"""Exercise the legacy model stream without loading GGUF models or GPUs."""
import importlib
import sys
import types
import unittest
from unittest.mock import Mock, patch

from llm.llmcli.providers.remote_http import RemoteHTTPProvider


class Response:
    def __init__(self, lines): self.lines = lines
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def raise_for_status(self): pass
    def iter_lines(self, decode_unicode=False): yield from self.lines


class LegacyNativeMetricsTests(unittest.TestCase):
    def test_provider_emits_native_eval_counters_and_true_finish_reason(self):
        perf = types.SimpleNamespace(n_p_eval=100, t_p_eval_ms=300., n_eval=12, t_eval_ms=500.)
        binding = types.SimpleNamespace(llama_perf_context_reset=Mock(), llama_perf_context=Mock(return_value=perf))
        fake = types.ModuleType('llama_cpp'); fake.Llama = object; fake.llama_chat_format = object(); fake.llama_cpp = binding
        latency = types.ModuleType('llm.llmcli.common.latency'); latency.Timer = object
        with patch.dict(sys.modules, {'llama_cpp': fake, 'llm.llmcli.common.latency': latency}):
            sys.modules.pop('llm.llmcli.providers.llamacpp', None)
            module = importlib.import_module('llm.llmcli.providers.llamacpp')
            model = module.LlamaCppProvider.__new__(module.LlamaCppProvider)
            model.system_prompt = None; model.max_new_tokens = 10; model.temperature = 0; model.top_p = 1
            model.llm = types.SimpleNamespace(_ctx=types.SimpleNamespace(ctx=object()),
                create_chat_completion=lambda **kw: iter([
                    {'choices': [{'delta': {'content': 'hello'}, 'finish_reason': None}]},
                    {'choices': [{'delta': {}, 'finish_reason': 'stop'}]},
                ]))
            events = list(model.stream_events('hi'))
            self.assertEqual(events[0], {'delta': 'hello'})
            self.assertEqual(events[-1]['finish_reason'], 'stop')
            self.assertEqual(events[-1]['metrics']['legacyBatchedEvalSeconds'], .3)
            self.assertEqual(events[-1]['metrics']['legacySingleEvalSeconds'], .5)
            self.assertNotIn('serverPrefillSeconds', events[-1]['metrics'])
            self.assertNotIn('serverPromptTokens', events[-1]['metrics'])
            binding.llama_perf_context_reset.assert_called_once()
        sys.modules.pop('llm.llmcli.providers.llamacpp', None)

    def test_client_propagates_metrics_only_on_confirmed_completion(self):
        seen = []
        good = Response([b'{"delta":"hello"}', b'{"done":true,"finish_reason":"stop","metrics":{"legacyBatchedEvalSeconds":0.3}}'])
        with patch('llm.llmcli.providers.remote_http.requests.post', return_value=good):
            self.assertEqual(''.join(RemoteHTTPProvider('http://test').stream_with_metrics('q', seen.append)), 'hello')
        self.assertEqual(seen, [{'legacyBatchedEvalSeconds': .3}])
        bad = Response([b'{"delta":"partial"}', b'{"done":true,"finish_reason":"length","metrics":{"legacyBatchedEvalSeconds":0.3}}'])
        seen.clear()
        with patch('llm.llmcli.providers.remote_http.requests.post', return_value=bad):
            with self.assertRaisesRegex(RuntimeError, 'length'):
                list(RemoteHTTPProvider('http://test').stream_with_metrics('q', seen.append))
        self.assertEqual(seen, [])

    def test_pipeline_forwards_metrics_from_the_current_stream(self):
        import threading, time
        from test_pipeline_failure import load_pipeline
        seen = []
        class Provider:
            def stream_with_metrics(self, prompt, on_server_metrics=None):
                yield 'An answer.'
                on_server_metrics({'legacyBatchedEvalSeconds': .3})
        playback = Mock(); playback.interrupted = threading.Event()
        args = types.SimpleNamespace(min_words=1, tts_enabled=False, mode='stream',
            teaching_style='default', chunk_first=80, chunk_mid=100, chunk_later=200,
            chunk_wait=1, browser_audio=True, voice_prompt=None,
            llm_model='llama3_8b_q4', thinking_enabled=False)
        active = threading.Event(); active.set()
        timer = types.SimpleNamespace(elapsed=lambda: 3., t0=time.perf_counter())
        load_pipeline()(transcript='test question', stt_time=0, total_timer=timer,
            llm=Provider(), tts_engine=None, retriever=None, args=args, playback=playback,
            pipeline_active=active, judge=None, event_cb=seen.append)
        self.assertTrue(any(e.get('data', {}).get('legacyBatchedEvalSeconds') == .3 for e in seen))

    def test_server_emits_error_without_false_done(self):
        import ast, asyncio, json, threading
        from pathlib import Path
        path = Path(__file__).resolve().parents[2] / 'app/remote_llm_server.py'
        tree = ast.parse(path.read_text(encoding='utf8'))
        fn = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'stream')
        fn.decorator_list = []
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn], type_ignores=[])
        def failed(prompt):
            yield {'delta': 'partial'}
            raise ValueError('internal error')
        env = {'json': json, 'LLM_LOCK': threading.Lock(),
               'LLM': types.SimpleNamespace(stream_events=failed),
               'StreamingResponse': lambda it, media_type: it}
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), env)
        request = types.SimpleNamespace(json=Mock())
        async def body(): return {'prompt': 'q'}
        request.json = body
        events = [json.loads(line) for line in asyncio.run(env['stream'](request))]
        self.assertEqual(events[0], {'delta': 'partial'})
        self.assertEqual(events[1], {'error': {'type': 'ValueError'}})
        self.assertFalse(any(event.get('done') for event in events))

    def test_diagnostic_reads_new_terminal_event(self):
        from scripts.context_diagnostics import history_retrieval_pilot
        lines = [b'{"delta":"hello"}', b'{"done":true,"finish_reason":"stop","metrics":{"legacyBatchedEvalSeconds":0.3}}']
        with patch.object(history_retrieval_pilot.requests, 'post', return_value=Response(lines)):
            result = history_retrieval_pilot.generate_legacy('http://test', 'q')
        self.assertTrue(result['completion_verified'])
        self.assertEqual(result['legacyBatchedEvalSeconds'], .3)
        self.assertNotIn('serverPrefillSeconds', result)

    def test_client_rejects_error_and_missing_done(self):
        for lines in ([b'{"error":{"type":"RuntimeError"}}'], [b'{"delta":"partial"}']):
            with patch('llm.llmcli.providers.remote_http.requests.post', return_value=Response(lines)):
                with self.assertRaises(RuntimeError):
                    list(RemoteHTTPProvider('http://test').stream('q'))

if __name__ == '__main__': unittest.main()
