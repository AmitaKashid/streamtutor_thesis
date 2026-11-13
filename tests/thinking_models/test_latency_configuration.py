import os
import unittest
from unittest.mock import patch
from llm.llmcli.providers.thinking_http import ThinkingHTTPProvider
from test_thinking_http_encoding import Response


class LatencyConfigurationTests(unittest.TestCase):
    def test_budget_is_sent_when_enabled(self):
        with patch.dict(os.environ, {'LLMTUTOR_THINKING_BUDGET': '512'}):
            with patch('llm.llmcli.providers.thinking_http.requests.post', return_value=Response()) as post:
                self.assertEqual(''.join(ThinkingHTTPProvider('http://test').stream('question')), 'Example 😊')
                self.assertEqual(post.call_args.kwargs['json']['reasoning_budget_tokens'], 512)

    def test_invalid_budget_fails_before_network(self):
        with patch.dict(os.environ, {'LLMTUTOR_THINKING_BUDGET': '0'}):
            with patch('llm.llmcli.providers.thinking_http.requests.post') as post:
                with self.assertRaises(ValueError):
                    list(ThinkingHTTPProvider('http://test').stream('question'))
                post.assert_not_called()

    def test_unset_budget_preserves_server_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch('llm.llmcli.providers.thinking_http.requests.post', return_value=Response()) as post:
                list(ThinkingHTTPProvider('http://test').stream('question'))
                self.assertNotIn('reasoning_budget_tokens', post.call_args.kwargs['json'])

class PerTurnBudgetTests(unittest.TestCase):
    def test_request_budget_overrides_environment_and_does_not_leak(self):
        with patch.dict(os.environ, {'LLMTUTOR_THINKING_BUDGET': '1024'}):
            with patch('llm.llmcli.providers.thinking_http.requests.post', side_effect=[Response(), Response()]) as post:
                provider = ThinkingHTTPProvider('http://test')
                list(provider.stream_with_cancel('first', None, reasoning_budget_tokens=256))
                list(provider.stream_with_cancel('second', None, reasoning_budget_tokens=512))
                self.assertEqual([call.kwargs['json']['reasoning_budget_tokens'] for call in post.call_args_list], [256, 512])

    def test_disabled_thinking_never_sends_budget(self):
        from test_thinking_toggle import Response as ToggleResponse
        response = ToggleResponse([({'content': 'answer'}, None), ({}, 'stop')])
        with patch.dict(os.environ, {'LLMTUTOR_THINKING_BUDGET': '512'}):
            with patch('llm.llmcli.providers.thinking_http.requests.post', return_value=response) as post:
                list(ThinkingHTTPProvider('http://test').stream_with_cancel('q', None, thinking_enabled=False, reasoning_budget_tokens=256))
                self.assertNotIn('reasoning_budget_tokens', post.call_args.kwargs['json'])

    def test_invalid_per_turn_budget_fails_before_network(self):
        with patch('llm.llmcli.providers.thinking_http.requests.post') as post:
            with self.assertRaises(ValueError):
                list(ThinkingHTTPProvider('http://test').stream_with_cancel('q', None, reasoning_budget_tokens=True))
            post.assert_not_called()
