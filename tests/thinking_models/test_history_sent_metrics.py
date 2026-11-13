"""Regression tests for the history actually included in each request."""
import threading
import time
import types
import unittest
from unittest.mock import Mock

from test_context_measurements import load_memory_class
from test_pipeline_failure import load_pipeline


class HistorySentMetricsTests(unittest.TestCase):
    def run_turn(self, memory, enabled):
        prompts, events = [], []
        class Provider:
            def stream(self, prompt):
                prompts.append(prompt)
                return iter(['A new answer.'])
        args = types.SimpleNamespace(min_words=1, tts_enabled=False, mode='stream',
            teaching_style='default', chunk_first=80, chunk_mid=100, chunk_later=200,
            chunk_wait=1, browser_audio=True, voice_prompt=None,
            llm_model='qwen', thinking_enabled=False, conversation_context=enabled)
        playback = Mock(); playback.interrupted = threading.Event()
        active = threading.Event(); active.set()
        timer = types.SimpleNamespace(elapsed=lambda: 3.0, t0=time.perf_counter())
        load_pipeline()(transcript='new question', stt_time=0, total_timer=timer,
            llm=Provider(), tts_engine=None, retriever=None, args=args, playback=playback,
            pipeline_active=active, judge=None, conversation_memory=memory, event_cb=events.append)
        metrics = [e['data'] for e in events if e['type'] == 'metrics' and 'conversationChars' in e.get('data', {})]
        self.assertEqual(len(metrics), 1)
        return prompts[0], metrics[0]

    def test_metric_describes_prompt_before_answer_is_stored(self):
        memory = load_memory_class()(max_turns=2)
        for i in range(3): memory.add_turn(f'old question {i}', f'old answer {i}')
        history = memory.format_history()
        prompt, metric = self.run_turn(memory, True)
        self.assertIn(history, prompt)
        self.assertNotIn('old question 0', prompt)
        self.assertEqual(metric['conversationChars'], len(history))
        self.assertEqual(metric['conversationTurns'], 2)
        self.assertNotEqual(metric['conversationChars'], len(memory.format_history()))
        self.assertIn('new question', memory.format_history())

    def test_disabled_context_reports_zero_even_when_memory_exists(self):
        memory = load_memory_class()()
        memory.add_turn('old question', 'old answer')
        prompt, metric = self.run_turn(memory, False)
        self.assertEqual(prompt, 'new question')
        self.assertEqual((metric['conversationChars'], metric['conversationTurns']), (0, 0))

    def test_truncation_counts_only_turns_with_text_remaining(self):
        memory = load_memory_class()(max_chars=12)
        memory.add_turn('first', 'first answer')
        memory.add_turn('second', 'last answer')
        history, count = memory.format_history_snapshot(memory.snapshot())
        prompt, metric = self.run_turn(memory, True)
        self.assertIn(history, prompt)
        self.assertEqual(count, 1)
        self.assertEqual(metric['conversationTurns'], 1)
        self.assertEqual(metric['conversationChars'], len(history))


if __name__ == '__main__':
    unittest.main()
