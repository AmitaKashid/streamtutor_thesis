import unittest
import threading
import time
import types
from unittest.mock import Mock
from app.followup_retrieval import followup_retrieval_query
from test_context_measurements import load_memory_class
from test_pipeline_failure import load_pipeline


class FollowupRetrievalTests(unittest.TestCase):
    def test_elliptical_question_uses_recent_student_topic_not_generated_answer(self):
        history = [
            {'user':'What is a decision tree?', 'assistant':'Fabricated patient dataset.'},
            {'user':'Could you explain that?', 'assistant':'Another invented example.'},
        ]
        query = followup_retrieval_query('Can you explain that with an example?', history)
        self.assertIn('Can you explain that with an example?', query)
        self.assertIn('What is a decision tree?', query)
        self.assertNotIn('patient', query)
        self.assertNotIn('invented', query)

    def test_new_topic_does_not_inherit_a_stale_topic(self):
        history = [{'user':'What is entropy?', 'assistant':'An answer.'}]
        for question in ('What is a decision tree?', 'Give an example of a decision tree.',
                         'How does that relate to classification?'):
            with self.subTest(question=question):
                self.assertEqual(followup_retrieval_query(question, history), question)

    def test_missing_topic_and_disabled_history_preserve_original(self):
        question = 'Can you explain that with an example?'
        self.assertEqual(followup_retrieval_query(question, []), question)
        self.assertEqual(followup_retrieval_query(question, [{'user':'Explain that again.'}]), question)

    def test_prior_user_question_is_bounded(self):
        history = [{'user':'decision tree ' + 'x' * 5000}]
        result = followup_retrieval_query('Explain that.', history)
        self.assertLessEqual(len(result), len('Explain that.') + 1 + 320)

    def test_real_pipeline_retrieves_with_the_same_history_used_in_the_prompt(self):
        memory = load_memory_class()()
        memory.add_turn('What is a decision tree?', 'A tree splits data.')
        queries, prompts = [], []
        class Retriever:
            def retrieve(self, query, top_k):
                queries.append((query, top_k))
                return []
            def format_context(self, hits): return ''
        class Provider:
            def stream(self, prompt):
                prompts.append(prompt)
                return iter(['Here is an example.'])
        args = types.SimpleNamespace(min_words=1, tts_enabled=False, mode='stream',
            teaching_style='default', chunk_first=80, chunk_mid=100, chunk_later=200,
            chunk_wait=1, browser_audio=True, voice_prompt=None, llm_model='llama3_8b_q4',
            thinking_enabled=False, conversation_context=True, rag_top_k=2,
            rag_print_context=False)
        playback = Mock(); playback.interrupted = threading.Event()
        active = threading.Event(); active.set()
        timer = types.SimpleNamespace(elapsed=lambda: 3., t0=time.perf_counter())
        load_pipeline()(transcript='Can you explain that with an example?', stt_time=0,
            total_timer=timer, llm=Provider(), tts_engine=None, retriever=Retriever(),
            args=args, playback=playback, pipeline_active=active, judge=None,
            conversation_memory=memory)
        self.assertEqual(queries[0][1], 2)
        self.assertIn('What is a decision tree?', queries[0][0])
        self.assertIn('What is a decision tree?', prompts[0])


if __name__ == '__main__': unittest.main()
