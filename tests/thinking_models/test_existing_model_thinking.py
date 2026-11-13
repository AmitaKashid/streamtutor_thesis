"""Routing and model identity checks for the existing thinking-capable models."""
import ast
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts import start_all_models as launcher
from llm.llmcli.providers.thinking_http import ThinkingHTTPProvider

ROOT = Path(__file__).resolve().parents[2]


def build_llm_function():
    tree = ast.parse((ROOT / 'app/mic_stt_llm_tts.py').read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'build_llm')
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    class LegacyProvider:
        def __init__(self, **kwargs): pass
        def healthcheck(self): return True
    env = {'RemoteHTTPProvider': LegacyProvider}
    exec(compile(module, '<build_llm>', 'exec'), env)
    return env['build_llm'], LegacyProvider


class ExistingModelThinkingTests(unittest.TestCase):
    def test_existing_thinking_models_use_reasoning_provider(self):
        build, legacy = build_llm_function()
        with patch.object(ThinkingHTTPProvider, 'healthcheck', return_value=True):
            for key in ('gemma4_26b_q3', 'qwen3_5_9b_q6'):
                with self.subTest(model=key):
                    args = types.SimpleNamespace(llm_backend='remote', llm_model=key,
                                                 llm_server_url='http://127.0.0.1:8007', llm_read_timeout=20)
                    self.assertIsInstance(build(args), ThinkingHTTPProvider)
            args = types.SimpleNamespace(llm_backend='remote', llm_model='llama3_8b_q4',
                                         llm_server_url='http://127.0.0.1:8000', llm_connect_timeout=1, llm_read_timeout=20)
            self.assertIsInstance(build(args), legacy)

    def test_existing_model_identity_rejects_other_gguf(self):
        for key, file in (('gemma4_26b_q3', 'gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf'),
                          ('qwen3_5_9b_q6', 'Qwen_Qwen3.5-9B-Q6_K.gguf')):
            with self.subTest(model=key):
                self.assertEqual(launcher.verify_model_identity(key, {'data': [{'id': '/models/' + file}]}), file)
                with self.assertRaises(RuntimeError):
                    launcher.verify_model_identity(key, {'data': [{'id': '/models/wrong.gguf'}]})

if __name__ == '__main__': unittest.main()
