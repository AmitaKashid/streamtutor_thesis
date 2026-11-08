"""Adapter tests with stubbed external provider: no GPU or remote endpoint required."""
import importlib
import sys
import types
import unittest
from unittest.mock import patch
from judge.runtime_service_v1 import ReferenceUnavailable
from judge.runtime_contract_v1 import EvaluationRequest, TurnOutcome, CONTRACT_VERSION


class StubJudge:
    def __init__(self): self.calls=[]
    def evaluate_dimension(self, **kwargs):
        self.calls.append(kwargs)
        return {"status":"valid", "result":{"score":4,"feedback":"Evidence supports the response"}}


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.original = sys.modules.pop('judge.runtime_scalar_adapter_v3',None)
        fake_scalar = types.ModuleType('judge.scalar_evaluator_v3')
        fake_scalar.ScalarJudgeV3 = object
        fake_scalar.load_frozen_rubric = lambda x: None
        fake_remote = types.ModuleType('llm.llmcli.providers.remote_http')
        fake_remote.RemoteHTTPProvider = object
        with patch.dict(sys.modules, {'judge.scalar_evaluator_v3':fake_scalar,
                                      'llm':types.ModuleType('llm'),
                                      'llm.llmcli':types.ModuleType('llm.llmcli'),
                                      'llm.llmcli.providers':types.ModuleType('llm.llmcli.providers'),
                                      'llm.llmcli.providers.remote_http':fake_remote}):
            self.module=importlib.import_module('judge.runtime_scalar_adapter_v3')
    def tearDown(self):
        sys.modules.pop('judge.runtime_scalar_adapter_v3',None)
        if self.original is not None: sys.modules['judge.runtime_scalar_adapter_v3']=self.original

    def request(self):
        return EvaluationRequest(
            contract_version=CONTRACT_VERSION,turn_id='t',conversation_id='c',
            outcome=TurnOutcome.DELIVERED,question_text='What is entropy?',
            generated_answer_text='Entropy measures uncertainty.',
            delivered_answer_text='Entropy measures uncertainty.',
            history=(),evidence=(),tutor_model_id='qwen',
            tutor_prompt_sha256='a'*64,rubric_sha256='b'*64,
            judge_model_id='fare_20b_q4',judge_prompt_sha256='c'*64,
            evaluation_dimension='correctness',teacher_style='socratic',
            generation_completed=True,delivery_completed=True)

    def test_missing_gold_skips_without_judge_call(self):
        m=StubJudge(); adapter=self.module.RuntimeScalarV3Adapter(m,{})
        with self.assertRaises(ReferenceUnavailable): adapter(self.request())
        self.assertEqual([],m.calls)

    def test_real_reference_passed_to_v3_evaluator(self):
        m=StubJudge(); adapter=self.module.RuntimeScalarV3Adapter(m,{
            'What is entropy?':'Entropy is a measure of uncertainty.'})
        self.assertEqual((4,'Evidence supports the response'),adapter(self.request()))
        self.assertEqual(m.calls[0]['reference_answer'],'Entropy is a measure of uncertainty.')
        self.assertEqual(m.calls[0]['dimension'],'correctness')

if __name__=='__main__': unittest.main()
