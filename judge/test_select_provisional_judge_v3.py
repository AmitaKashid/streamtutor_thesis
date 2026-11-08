from __future__ import annotations
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
from judge.select_provisional_judge_v3 import analyze, main

class ProvisionalJudgeDecisionV3Tests(unittest.TestCase):
    def test_frozen_input_analysis(self):
        source=Path(__file__).resolve().parents[1] / 'day4_task2_frozen_inputs.zip'
        if not source.exists():
            self.skipTest('Frozen evidence archive not present (available as separate local file)')
        result=analyze(source)
        self.assertEqual(sum(x['total_recorded'] for x in result['metrics'].values()),4300)
        self.assertEqual(result['selected_scalar_judge'],'fare_20b_q4')
        self.assertEqual(result['fallback_scalar_judge'],'compassjudger_2_7b_q4')
        self.assertEqual(result['metrics']['prometheus_7b_v2_q4']['contrast_bidirectional_pairs'],3)
        self.assertEqual(result['metrics']['fare_20b_q4']['critical_risks']['groundedness']['high_4_or_5'],0)
    def test_missing_archive_is_error(self):
        with self.assertRaises(FileNotFoundError):
            analyze(Path('__missing_archive__'))

if __name__=='__main__':
    unittest.main()
