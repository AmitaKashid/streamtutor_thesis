import unittest
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch
from scripts.context_diagnostics import history_retrieval_pilot as p

class RetrievalHistoryTests(unittest.TestCase):
    def test_counterfactual_windows_change_only_at_oldest_turn(self):
        a, b = p.build_case(75, 25), p.build_case(60, 40)
        self.assertEqual(a['user_utterance'], b['user_utterance'])
        self.assertEqual(a['conversation_history'][2:], b['conversation_history'][2:])
        jobs = p.make_schedule('fixed slide', 42)
        self.assertEqual(len(jobs), 8)
        lookup = {(c['question_id'], n): prompt for c, n, prompt, actual in jobs}
        for n in (0, 1, 3):
            self.assertEqual(lookup[('synthetic-gini-75-25', n)], lookup[('synthetic-gini-60-40', n)])
        self.assertNotEqual(lookup[('synthetic-gini-75-25', 5)], lookup[('synthetic-gini-60-40', 5)])

    def test_full_pilot_uses_one_retrieval_and_records_eight_responses(self):
        hit = types.SimpleNamespace(slide_id='slide', source_pdf='course.pdf', slide_number=1, score=0.8)
        class Retriever:
            calls = 0
            def __init__(self, config): pass
            def retrieve(self, query, top_k):
                self.__class__.calls += 1
                return [hit]
            def format_context(self, hits): return 'Fixed slide context.'
        fake_module = types.ModuleType('rag.rag_retrieve'); fake_module.RagRetriever = Retriever
        result = {'answer': 'example', 'stream_done': True, 'finish_reason': 'stop',
                  'unexpected_reasoning': False, 'serverPromptTokens': 10, 'serverCachedTokens': 0,
                  'serverPrefillSeconds': .1, 'serverOutputTokens': 2, 'serverDecodeSeconds': .2}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'out.jsonl'
            config = Path(directory) / 'config.yaml'; config.write_text('top_k: 2', encoding='utf8')
            argv = ['pilot', '--url', 'http://test', '--model-label', 'test', '--output', str(output), '--rag-config', str(config)]
            with patch.dict(sys.modules, {'rag.rag_retrieve': fake_module}), \
                 patch.object(sys, 'argv', argv), patch.object(p, 'generate', return_value=result) as generate:
                self.assertEqual(p.main(), 0)
            records = [json.loads(line) for line in output.read_text(encoding='utf8').splitlines()]
        self.assertEqual(Retriever.calls, 1)
        self.assertEqual(generate.call_count, 8)
        self.assertEqual(len(records), 8)
        self.assertEqual(len({r['rag_sha256'] for r in records}), 1)
        self.assertTrue(all(r['complete'] and r['metrics_complete'] for r in records))

    def test_rag_reference_detector_is_conservative(self):
        self.assertTrue(p.has_explicit_reference('Class is 75 percent yes', [75, 25]))
        self.assertTrue(p.has_explicit_reference('25% no', [75, 25]))
        self.assertFalse(p.has_explicit_reference('75th slide', [75, 25]))

if __name__ == '__main__': unittest.main()
