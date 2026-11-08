import tempfile, time, unittest
from pathlib import Path
from dataclasses import replace
from judge.runtime_service_v1 import RuntimeJudgeService
from judge.test_runtime_contract_v1 import request
from judge.runtime_contract_v1 import TurnOutcome

class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.db=Path(self.tmp.name)/'spool.db'
        self.services=[]
    def tearDown(self):
        for s in self.services: s.close()
        self.tmp.cleanup()
    def make(self, fn, **kw):
        s=RuntimeJudgeService(self.db,fn,**kw);self.services.append(s);return s
    def wait(self,s,key):
        for _ in range(200):
            row=s.read(key)
            if row and row['status'] in ('valid','invalid','error','skipped'): return row
            time.sleep(.01)
        self.fail('worker did not finish')
    def test_success_and_dedup(self):
        calls=[]
        s=self.make(lambda r:(calls.append(r.turn_id) or 4,'Good'))
        r=request()
        key,state=s.submit(r);self.assertEqual(state,'queued')
        self.assertEqual(self.wait(s,key)['score'],4)
        self.assertEqual(s.submit(r),(key,'valid'))
        self.assertEqual(len(calls),1)
    def test_payload_conflict(self):
        s=self.make(lambda r:(3,'Good'))
        s.submit(request())
        with self.assertRaises(ValueError): s.submit(replace(request(), question_text='changed'))
    def test_skipped_interruption(self):
        s=self.make(lambda r: self.fail('must not evaluate'))
        k,status=s.submit(request(outcome=TurnOutcome.INTERRUPTED,delivery_completed=False,delivered_answer_text='partial'))
        self.assertEqual(status,'skipped');self.assertEqual(s.read(k)['reason'],'interrupted')
    def test_queue_full(self):
        s=self.make(lambda r:(4,'Good'),autostart=False,queue_limit=1)
        self.assertEqual(s.submit(request())[1],'queued')
        k,status=s.submit(request(turn_id='other'))
        self.assertEqual(status,'skipped');self.assertEqual(s.read(k)['reason'],'queue_full')
    def test_retries_and_null_score(self):
        count=[]
        def bad(r): count.append(1); raise ValueError('bad scalar output')
        s=self.make(bad,max_attempts=2)
        k,_=s.submit(request())
        row=self.wait(s,k)
        self.assertEqual(row['status'],'invalid');self.assertIsNone(row['score']);self.assertEqual(row['attempts'],2)
    def test_resume_pending(self):
        s=self.make(lambda r:(4,'Good'),autostart=False)
        k,_=s.submit(request())
        s2=self.make(lambda r:(5,'Recovered'))
        self.assertEqual(self.wait(s2,k)['score'],5)
    def test_no_grounding_without_evidence(self):
        s=self.make(lambda r: self.fail('should skip'))
        k,status=s.submit(request(evidence=(),evaluation_dimension='groundedness'))
        self.assertEqual(status,'skipped');self.assertEqual(s.read(k)['reason'],'no_retrieved_evidence')
if __name__=='__main__': unittest.main()
