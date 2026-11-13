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
