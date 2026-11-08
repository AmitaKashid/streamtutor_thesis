import copy
import unittest
from pathlib import Path

from judge.analyze_judge_selection_v3 import load_runs, audit, triples, contrast_summary, paired_cluster_bootstrap

ARCHIVE = Path(__file__).resolve().parents[1] / 'day4_task2_frozen_inputs.zip'

@unittest.skipUnless(ARCHIVE.exists(), "Optional frozen evidence ZIP not present; run integration tests with archive")
class FrozenJudgeSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs=load_runs(ARCHIVE)
        cls.frame,cls.run_metrics,cls.provenance=audit(cls.runs)
    def test_frozen_matrix(self):
        self.assertEqual(len(self.runs),16)
        self.assertEqual(len(self.frame),4300)
        self.assertEqual(int(self.run_metrics.invalid.sum()),17)
    def test_missing_record_rejected(self):
        x=copy.deepcopy(self.runs)
        key=next(k for k in x if k[1]=='selection')
        x[key][1].pop()
        with self.assertRaises(ValueError):audit(x)
    def test_changed_hash_rejected(self):
        x=copy.deepcopy(self.runs)
        key=next(k for k in x if k[1]=='selection')
        x[key][1][0]['input_sha256']='bad'
        with self.assertRaises(ValueError):audit(x)
    def test_duplicate_record_rejected(self):
        x=copy.deepcopy(self.runs)
        key=next(k for k in x if k[1]=='selection')
        x[key][1][1]['evaluation_id']=x[key][1][0]['evaluation_id']
        with self.assertRaises(ValueError):audit(x)
    def test_triple_counts_and_paired_ci(self):
        t=triples(self.frame,'selection')
        self.assertEqual(len(t),1007)
        b=paired_cluster_bootstrap(t,nboot=20)
        self.assertEqual(len(b),6)
        self.assertTrue((b.common_triples>0).all())
    def test_contrasts_invalid_not_dropped(self):
        c=contrast_summary(self.frame).set_index('model')
        self.assertEqual(int(c.loc['prometheus_7b_v2_q4','valid_judgments']),13)
        self.assertEqual(int(c.loc['prometheus_7b_v2_q4','complete_pairs']),3)
if __name__=='__main__':unittest.main()
