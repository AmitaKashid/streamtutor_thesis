"""Test how retained and truncated conversation boundaries reach the model."""
import threading
import time
import types
import unittest
from unittest.mock import Mock

from test_context_measurements import load_memory_class
from test_pipeline_failure import load_pipeline


class HistoryBoundaryTests(unittest.TestCase):
    def test_discarded_turn_remains_known_after_limit_increases(self):
        memory = load_memory_class()(max_turns=2)
        for i in range(3): memory.add_turn(f'question {i}', f'answer {i}')
        memory.max_turns = 10
        turns, omitted = memory.snapshot_with_omission()
        self.assertTrue(omitted)
        self.assertEqual([x['user'] for x in turns], ['question 1', 'question 2'])
        prompt = memory.build_contextual_user_query('What was first?')
        self.assertIn('history below is incomplete', prompt)
        self.assertIn('turn numbers refer only to the excerpt', prompt)
        self.assertNotIn('question 0', prompt)

    def test_temporarily_lowered_limit_and_full_history(self):
        memory = load_memory_class()()
        for i in range(3): memory.add_turn(f'q{i}', f'a{i}')
        self.assertNotIn('history below is incomplete', memory.build_contextual_user_query('q'))
        memory.max_turns = 1
        self.assertTrue(memory.snapshot_with_omission()[1])
        self.assertIn('history below is incomplete', memory.build_contextual_user_query('q'))
        memory.max_turns = None
        self.assertFalse(memory.snapshot_with_omission()[1])

    def test_character_truncation_is_marked(self):
        memory = load_memory_class()(max_chars=5)
        memory.add_turn('the question', 'the answer')
        self.assertIn('history below is incomplete', memory.build_contextual_user_query('q'))

    def test_pipeline_sends_omission_message_and_unmodified_metrics(self):
        memory = load_memory_class()(max_turns=1)
        memory.add_turn('first question', 'first answer')
        memory.add_turn('second question', 'second answer')
        history = memory.format_history()
        prompts, events = [], []
        class Provider:
            def stream(self, prompt):
                prompts.append(prompt)
                return iter(['A complete answer.'])
        args = types.SimpleNamespace(min_words=1, tts_enabled=False, mode='stream',
            teaching_style='default', chunk_first=80, chunk_mid=100, chunk_later=200,
            chunk_wait=1, browser_audio=True, voice_prompt=None,
            llm_model='qwen', thinking_enabled=False, conversation_context=True)
        playback = Mock(); playback.interrupted = threading.Event()
        active = threading.Event(); active.set()
        timer = types.SimpleNamespace(elapsed=lambda: 3.0, t0=time.perf_counter())
        load_pipeline()(transcript='What was first?', stt_time=0, total_timer=timer,
            llm=Provider(), tts_engine=None, retriever=None, args=args, playback=playback,
            pipeline_active=active, judge=None, conversation_memory=memory, event_cb=events.append)
        self.assertIn('history below is incomplete', prompts[0])
        self.assertIn(history, prompts[0])
        metrics = next(e['data'] for e in events if e['type'] == 'metrics' and 'conversationChars' in e.get('data', {}))
        self.assertEqual(metrics['conversationChars'], len(history))
        self.assertEqual(metrics['conversationTurns'], 1)


if __name__ == '__main__': unittest.main()
