"""Reject healthy but incorrectly routed or ambiguous evaluation endpoints."""
import unittest
from unittest.mock import patch
from scripts.context_diagnostics.verify_eval_endpoint import check_endpoint, EXPECTED


class Response:
    def __init__(self, data): self.data = data
    def raise_for_status(self): pass
    def json(self): return self.data


class EvalEndpointTests(unittest.TestCase):
    def test_accepts_expected_reported_model_path(self):
        with patch('scripts.context_diagnostics.verify_eval_endpoint.requests.get', side_effect=[
            Response({'status': 'ok'}), Response({'data': [{'id': '/models/' + EXPECTED}]})]):
            self.assertEqual(check_endpoint('http://test'), ['/models/' + EXPECTED])

    def test_rejects_wrong_or_ambiguous_model(self):
        for rows in ([], [{'id': 'different.gguf'}], [{'id': EXPECTED}, {'id': 'other.gguf'}]):
            with self.subTest(rows=rows), patch('scripts.context_diagnostics.verify_eval_endpoint.requests.get',
                    side_effect=[Response({'status': 'ok'}), Response({'data': rows})]):
                with self.assertRaises(ValueError): check_endpoint('http://test')

    def test_rejects_unready_server(self):
        with patch('scripts.context_diagnostics.verify_eval_endpoint.requests.get', return_value=Response({'status': 'loading'})):
            with self.assertRaises(ValueError): check_endpoint('http://test')

if __name__ == '__main__': unittest.main()
