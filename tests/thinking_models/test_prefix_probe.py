import unittest
from unittest.mock import patch
from scripts.context_diagnostics import prefix_probe


class PrefixProbeTests(unittest.TestCase):
    def test_shared_prefix_stops_at_first_mismatch(self):
        self.assertEqual(prefix_probe.shared_prefix([1, 2, 3, 4], [1, 2, 9, 4]), 2)
        self.assertEqual(prefix_probe.shared_prefix([1], [1, 2]), 1)

    def test_separate_primes_and_reported_counts(self):
        results = [dict(serverCachedTokens=v, serverPromptTokens=3) for v in (0, 3, 3, 1)]
        with patch.object(prefix_probe, 'tokenized_prompt', side_effect=[[1, 2, 3], [1, 7, 8]]), \
             patch.object(prefix_probe, 'generate', side_effect=results) as calls:
            row = prefix_probe.run_condition('http://server', 3, 0)
        self.assertEqual(row['shared_prefix_tokens'], 1)
        self.assertEqual(row['identical']['serverCachedTokens'], 3)
        self.assertEqual(row['shifted']['serverCachedTokens'], 1)
        self.assertTrue(row['token_count_matches_server'])
        self.assertEqual(calls.call_count, 4)
        self.assertEqual(calls.call_args_list[0], calls.call_args_list[2])
        self.assertNotEqual(calls.call_args_list[2], calls.call_args_list[3])

if __name__ == '__main__':
    unittest.main()
