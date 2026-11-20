"""Exercise both reviewed llama-cpp-python streaming interfaces without a GPU."""
import importlib
import sys
import types
import unittest
from unittest.mock import Mock, patch


class FakeLlama:
    def __init__(self, cached=2, fail=False, tokens=(11, 12, 13, 999)):
        self.cached = cached
        self.fail = fail
        self.tokens = tokens
        self.draft_model = None
        self._ctx = types.SimpleNamespace(ctx=object())
        self._model = types.SimpleNamespace(model=object(), vocab=object())

    def eval(self, tokens):
        if self.fail:
            raise RuntimeError('inference failed')

    def generate(self, tokens, **kwargs):
        self.eval(tokens[self.cached:])
        for index, token in enumerate(self.tokens):
            if index:
                self.eval([self.tokens[index - 1]])
            yield token

    def create_chat_completion(self, **kwargs):
        for token in self.generate([1, 2, 3, 4, 5]):
            if token == 999:
                break
            yield {'choices': [{'delta': {'content': str(token)}, 'finish_reason': None}]}
        yield {'choices': [{'delta': {}, 'finish_reason': 'stop'}]}


class Llama3StreamMetricsTests(unittest.TestCase):
    def setUp(self):
        self.binding = types.SimpleNamespace(
            llama_token_is_eog=Mock(side_effect=lambda model, token: token == 999),
            llama_vocab_is_eog=Mock(side_effect=lambda vocab, token: token == 999),
            llama_synchronize=Mock(),
            llama_perf_context_reset=Mock(),
            llama_perf_context=Mock(side_effect=AttributeError()),
        )
        fake = types.ModuleType('llama_cpp')
        fake.__version__ = '0.3.4'
        fake.Llama = object
        fake.llama_chat_format = object()
        fake.llama_cpp = self.binding
        latency = types.ModuleType('llm.llmcli.common.latency')
        latency.Timer = object
        self.modules = patch.dict(sys.modules, {'llama_cpp': fake, 'llm.llmcli.common.latency': latency})
        self.modules.start()
        sys.modules.pop('llm.llmcli.providers.llamacpp', None)
        self.module = importlib.import_module('llm.llmcli.providers.llamacpp')
        self.package = fake

    def tearDown(self):
        sys.modules.pop('llm.llmcli.providers.llamacpp', None)
        self.modules.stop()

    def provider(self, llm):
        provider = self.module.LlamaCppProvider.__new__(self.module.LlamaCppProvider)
        provider.system_prompt = None
        provider.max_new_tokens = 20
        provider.temperature = 0
        provider.top_p = 1
        provider.chat_format = 'llama-3'
        provider.llm = llm
        return provider

    def test_both_versions_measure_actual_prompt_cache_and_output(self):
        for version in ('0.3.4', '0.3.20'):
            with self.subTest(version=version):
                self.package.__version__ = version
                llm = FakeLlama()
                events = list(self.provider(llm).stream_events('question'))
                self.assertEqual(''.join(e.get('delta', '') for e in events), '111213')
                metrics = events[-1]['metrics']
                self.assertEqual(metrics['serverPromptTokens'], 5)
                self.assertEqual(metrics['serverProcessedPromptTokens'], 3)
                self.assertEqual(metrics['serverCachedTokens'], 2)
                self.assertEqual(metrics['serverOutputTokens'], 3)
                self.assertGreaterEqual(metrics['serverPrefillSeconds'], 0)
                self.assertGreater(metrics['serverDecodeSeconds'], 0)
                self.assertGreater(metrics['serverDecodeTokensPerSecond'], 0)
                self.assertEqual(metrics['serverMetricsSource'], 'llama_cpp_python_observed_v1')
                self.assertEqual(events[-1]['finish_reason'], 'stop')
                self.assertNotIn('eval', llm.__dict__)
                self.assertNotIn('generate', llm.__dict__)
        self.assertGreaterEqual(self.binding.llama_token_is_eog.call_count, 4)
        self.assertGreaterEqual(self.binding.llama_vocab_is_eog.call_count, 4)

    def test_unknown_version_and_other_models_do_not_install_hooks(self):
        self.package.__version__ = '0.99.0'
        llm = FakeLlama()
        metrics = list(self.provider(llm).stream_events('q'))[-1]['metrics']
        self.assertNotIn('serverPromptTokens', metrics)
        self.assertNotIn('serverOutputTokens', metrics)
        self.assertNotIn('eval', llm.__dict__)
        self.package.__version__ = '0.3.4'
        provider = self.provider(llm)
        provider.chat_format = 'chatml'
        self.assertNotIn('serverPromptTokens', list(provider.stream_events('q'))[-1]['metrics'])

    def test_one_token_has_no_fabricated_decode_rate(self):
        metrics = list(self.provider(FakeLlama(cached=0, tokens=(11, 999))).stream_events('q'))[-1]['metrics']
        self.assertEqual(metrics['serverCachedTokens'], 0)
        self.assertEqual(metrics['serverOutputTokens'], 1)
        self.assertNotIn('serverDecodeSeconds', metrics)
        self.assertNotIn('serverDecodeTokensPerSecond', metrics)

    def test_inference_error_and_stream_close_restore_hooks(self):
        failed = FakeLlama(fail=True)
        with self.assertRaisesRegex(RuntimeError, 'inference failed'):
            list(self.provider(failed).stream_events('q'))
        self.assertNotIn('generate', failed.__dict__)
        self.assertNotIn('eval', failed.__dict__)
        cancelled = FakeLlama()
        stream = self.provider(cancelled).stream_events('q')
        next(stream)
        stream.close()
        self.assertNotIn('generate', cancelled.__dict__)
        self.assertNotIn('eval', cancelled.__dict__)

    def test_existing_instance_hooks_are_restored(self):
        llm = FakeLlama()
        original_generate, original_eval = llm.generate, llm.eval
        llm.generate, llm.eval = original_generate, original_eval
        list(self.provider(llm).stream_events('q'))
        self.assertIs(llm.generate, original_generate)
        self.assertIs(llm.eval, original_eval)

    def test_missing_synchronization_keeps_prefill_unavailable(self):
        self.binding.llama_synchronize = None
        metrics = list(self.provider(FakeLlama()).stream_events('q'))[-1]['metrics']
        self.assertEqual(metrics['serverPromptTokens'], 5)
        self.assertNotIn('serverPrefillSeconds', metrics)


if __name__ == '__main__':
    unittest.main()
