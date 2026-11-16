"""Metrics emitted by the real pipeline must retain their request identity."""
import threading
import time
import types
import unittest
from unittest.mock import Mock
from test_pipeline_failure import load_pipeline
from test_context_measurements import load_memory_class


class TurnMetricIdentityTest(unittest.TestCase):
    def test_empty_retrieval_still_emits_measured_rag_latency(self):
        class Provider:
            def stream(self, prompt):
                return iter(['An answer.'])
        retriever = types.SimpleNamespace(retrieve=lambda query, top_k: [],
                                          format_context=lambda hits: '')
        args = types.SimpleNamespace(min_words=1, tts_enabled=False, mode='stream',
            teaching_style='default', chunk_first=80, chunk_mid=100, chunk_later=200,
            chunk_wait=1, browser_audio=True, voice_prompt=None, stt_backend='test',
            llm_model='llama', thinking_enabled=False, conversation_context=False,
            rag_top_k=2, rag_print_context=False)
        playback = Mock(); playback.interrupted = threading.Event()
        active = threading.Event(); active.set(); events = []
        timer = types.SimpleNamespace(elapsed=lambda: 3.0, t0=time.perf_counter())
        load_pipeline()(transcript='new question', stt_time=.2, total_timer=timer,
            llm=Provider(), tts_engine=None, retriever=retriever, args=args,
            playback=playback, pipeline_active=active, judge=None,
            evaluation_turn_id='empty-rag', event_cb=events.append)
        measurements = [event for event in events if event['type'] == 'metrics'
                        and 'ragLatency' in event['data']]
        self.assertEqual(len(measurements), 1)
        self.assertEqual(measurements[0]['turn_id'], 'empty-rag')
        self.assertGreaterEqual(measurements[0]['data']['ragLatency'], 0)

    def test_success_and_interrupted_turns_are_tagged(self):
        memory = load_memory_class()()
        memory.add_turn('previous question', 'previous answer')
        args = types.SimpleNamespace(min_words=1, tts_enabled=False, mode='stream',
            teaching_style='default', chunk_first=80, chunk_mid=100, chunk_later=200,
            chunk_wait=1, browser_audio=True, voice_prompt=None, stt_backend='test',
            llm_model='qwen', thinking_enabled=False, conversation_context=True)
        for turn_id, fail in [('interrupted', True), ('next', False)]:
            events = []
            class Provider:
                def stream(self, prompt):
                    if fail:
                        raise RuntimeError('interrupted request')
                    return iter(['An answer.'])
            playback = Mock(); playback.interrupted = threading.Event()
            active = threading.Event(); active.set()
            timer = types.SimpleNamespace(elapsed=lambda: 3.0, t0=time.perf_counter())
            try:
                load_pipeline()(transcript='new question', stt_time=.25, total_timer=timer,
                    llm=Provider(), tts_engine=None, retriever=None, args=args, playback=playback,
                    pipeline_active=active, judge=None, conversation_memory=memory,
                    evaluation_turn_id=turn_id, event_cb=events.append)
            except RuntimeError:
                if not fail: raise
            metrics = [e for e in events if e['type'] == 'metrics']
            self.assertTrue(metrics)
            self.assertTrue(all(e.get('turn_id') == turn_id for e in metrics))
            self.assertEqual(next(e['data']['stt'] for e in metrics if 'stt' in e['data']), .25)
            self.assertTrue(any(e['data'].get('conversationTurns') == 1 for e in metrics))

if __name__ == '__main__': unittest.main()
