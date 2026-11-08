import tempfile,time,unittest
from pathlib import Path
from judge.runtime_bridge_v1 import Candidate, DeliveryCoordinator
from judge.runtime_service_v1 import RuntimeJudgeService
from judge.runtime_contract_v1 import digest_text

class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.calls=[]
        self.service=RuntimeJudgeService(Path(self.tmp.name)/'db.sqlite',lambda r:(self.calls.append(r.evaluation_dimension) or 4,'Valid'))
        self.bridge=DeliveryCoordinator(self.service,rubric_sha256='rubric',judge_prompt_sha256='prompt')
        self.c=Candidate('turn','session','What is entropy?','Entropy measures uncertainty.',(),
                         'From course slides: entropy measures uncertainty.','tutor-model',digest_text('prompt'),'socratic',True)
    def tearDown(self):self.service.close();self.tmp.cleanup()
    def test_no_judge_before_ack(self):
        self.bridge.candidate_ready(self.c)
        time.sleep(.03)
        self.assertEqual(self.calls,[])
    def test_ack_once_seven_dimensions(self):
        self.bridge.candidate_ready(self.c)
        self.assertEqual(len(self.bridge.acknowledge('turn',delivered=True)),7)
        self.assertEqual(self.bridge.acknowledge('turn',delivered=True),[])
    def test_interrupted_skips(self):
        self.bridge.candidate_ready(self.c)
        rs=self.bridge.acknowledge('turn',delivered=False)
        self.assertTrue(all(status=='skipped' for _,status in rs))
        self.assertEqual(self.calls,[])
    def test_no_rag_skips_groundedness(self):
        from dataclasses import replace
        c=replace(self.c,rag_context='')
        self.bridge.candidate_ready(c)
        outcomes=self.bridge.acknowledge('turn',delivered=True)
        self.assertEqual(sum(status=='skipped' for _,status in outcomes),1)
if __name__=='__main__': unittest.main()
