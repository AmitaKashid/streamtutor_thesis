"""Conversation history must reflect browser delivery, including interruptions."""
import threading
import time
import types
import unittest
from unittest.mock import Mock

from app.conversation_delivery import ConversationDelivery
from test_context_measurements import load_memory_class
from test_pipeline_failure import load_pipeline
from test_speech_text_segments import run_pipeline, StreamingTTS


class ConversationDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.memory = load_memory_class()()
        self.delivery = ConversationDelivery(self.memory)

    def test_interruption_does_not_store_generated_future_text(self):
        self.delivery.start('one', 'What is a decision tree?', True)
        self.delivery.expect('one', 'A tree splits data. ')
        self.delivery.expect('one', 'Imagine a patient dataset. ')
        self.assertTrue(self.delivery.progress('one', 'A tree splits data. '))
        self.delivery.finish('one', delivered=False)
        saved = self.memory.snapshot()[0]
        self.assertEqual(saved['user'], 'What is a decision tree?')
        self.assertIn('A tree splits data.', saved['assistant'])
        self.assertNotIn('patient', saved['assistant'])
        self.assertIn('partially displayed', saved['assistant'])

    def test_completed_turn_is_saved_only_after_browser_ack(self):
        self.delivery.start('one', 'Question?', True)
        self.delivery.expect('one', 'Answer. ')
        self.delivery.progress('one', 'Answer. ')
        self.assertEqual(self.memory.snapshot(), [])
        self.delivery.finish('one', delivered=True)
        self.delivery.finish('one', delivered=False)  # duplicate is harmless
        self.assertEqual(self.memory.snapshot(), [{'user':'Question?', 'assistant':'Answer.'}])

    def test_untrusted_duplicate_late_and_wrong_turn_progress_is_rejected(self):
        self.delivery.start('one', 'Question?', True)
        self.delivery.expect('one', 'First second. ')
        self.assertTrue(self.delivery.progress('one', 'First '))
        self.assertFalse(self.delivery.progress('one', 'First '))
        self.assertFalse(self.delivery.progress('other', 'second. '))
        self.assertFalse(self.delivery.progress('one', 'invented '))
        self.delivery.finish('one', delivered=False)
        self.assertFalse(self.delivery.progress('one', 'second. '))
        self.assertEqual(self.delivery.visible_chunks('one'), [])
        self.assertIn('First', self.memory.snapshot()[0]['assistant'])
        self.assertNotIn('second', self.memory.snapshot()[0]['assistant'])

    def test_pause_snapshot_excludes_words_that_arrived_after_speech_onset(self):
        self.delivery.start('one', 'Question?', True)
        self.delivery.expect('one', ' First second. ')
        self.delivery.progress('one', ' First ')
        snapshot = self.delivery.visible_chunks('one')
        self.delivery.progress('one', 'second. ')
        self.delivery.finish('one', delivered=False, visible_at_stop=snapshot)
        self.assertNotIn('second', self.memory.snapshot()[0]['assistant'])

    def test_empty_interruption_preserves_user_question_honestly(self):
        self.delivery.start('one', 'What is entropy?', True)
        self.delivery.expect('one', 'Unseen answer')
        self.delivery.finish('one', delivered=False)
        saved = self.memory.snapshot()[0]
        self.assertEqual(saved['user'], 'What is entropy?')
        self.assertIn('before any text was displayed', saved['assistant'])
        self.assertNotIn('Unseen answer', saved['assistant'])

    def test_pipeline_defers_browser_history_until_delivery_ack(self):
        events = run_pipeline('stream-input', StreamingTTS(), pieces=['Hello.'], immediate=True,
                              conversation_memory=self.memory, defer_browser_memory=True)
        self.assertIn('audio_done', [event['type'] for event in events])
        self.assertEqual(self.memory.snapshot(), [])
        self.delivery.start('turn-1', 'test question', True)
        for event in events:
            if event['type'] == 'audio_segment_start':
                self.delivery.expect('turn-1', event['text'])
        self.delivery.progress('turn-1', 'Hello. ')
        self.delivery.finish('turn-1', delivered=True)
        self.assertEqual(self.memory.snapshot()[0]['assistant'], 'Hello.')

    def test_pipeline_preserves_real_interruption_as_user_message(self):
        run = load_pipeline()
        run.__globals__['build_interrupt_user_query'] = lambda **kw: 'synthetic wrapper ' + kw['interruption_text']
        memory = self.memory
        memory.add_turn('first', 'shown')
        class Provider:
            def stream(self, prompt): return iter(['A complete answer.'])
        args = types.SimpleNamespace(min_words=1, tts_enabled=False, mode='stream',
            teaching_style='default', chunk_first=80, chunk_mid=100, chunk_later=200,
            chunk_wait=1, browser_audio=True, voice_prompt=None, llm_model='llama3_8b_q4',
            thinking_enabled=False, conversation_context=True)
        playback = Mock(); playback.interrupted = threading.Event()
        active = threading.Event(); active.set()
        timer = types.SimpleNamespace(elapsed=lambda: 3., t0=time.perf_counter())
        run(transcript='Can you explain that?', interruption_text='Can you explain that?',
            stt_time=0, total_timer=timer, llm=Provider(), tts_engine=None,
            retriever=None, args=args, playback=playback, pipeline_active=active,
            judge=None, conversation_memory=memory)
        self.assertEqual(memory.snapshot()[-1]['user'], 'Can you explain that?')
        self.assertNotIn('synthetic wrapper', memory.snapshot()[-1]['user'])


if __name__ == '__main__': unittest.main()
