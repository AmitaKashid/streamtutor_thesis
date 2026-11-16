"""Text boundaries must be safe across arbitrary LLM token events."""
import unittest
from unittest.mock import patch

from llm.llmcli.common.prosody_chunker import ProsodyChunker, ProsodyChunkerConfig


class SpokenPhraseBoundaryTests(unittest.TestCase):
    def make_chunker(self, **kwargs):
        config = dict(first_target_chars=60, later_target_chars=80,
                      first_min_sentence_chars=20, min_sentence_chars=20,
                      min_comma_chars=35, max_chars=120, max_wait_s=100.)
        config.update(kwargs)
        return ProsodyChunker(ProsodyChunkerConfig(**config))

    def test_decimal_abbreviation_and_url_are_not_cut_at_a_period(self):
        chunks = self.make_chunker()
        text = 'This probability equals 3.14 and Dr. Taylor uses U.S.A. data from example.com. Next sentence.'
        with patch('llm.llmcli.common.prosody_chunker.time.perf_counter', return_value=0):
            output = list(chunks.feed(list(text)))
            output.append(chunks.flush())
        self.assertEqual(output, ['This probability equals 3.14 and Dr. Taylor uses U.S.A. data from example.com.', 'Next sentence.'])

    def test_punctuation_split_across_model_events_waits_for_word_boundary(self):
        chunker = self.make_chunker(first_min_sentence_chars=10)
        with patch('llm.llmcli.common.prosody_chunker.time.perf_counter', return_value=0):
            self.assertEqual(list(chunker.feed(['Probability is 3.'])), [])
            self.assertEqual(list(chunker.feed(['14 for this example.', ' '])), ['Probability is 3.14 for this example.'])

    def test_force_flush_never_cuts_a_long_word(self):
        chunker = self.make_chunker(first_target_chars=100, max_chars=14, max_wait_s=0)
        with patch('llm.llmcli.common.prosody_chunker.time.perf_counter', return_value=1):
            self.assertEqual(list(chunker.feed(['Read ', 'extraordinarilylongtechnicalword'])), [])
            self.assertEqual(list(chunker.feed([' next '])), ['Read extraordinarilylongtechnicalword'])
            self.assertEqual(chunker.flush(), 'next')

    def test_closing_quote_remains_on_the_sentence(self):
        chunker = self.make_chunker(first_min_sentence_chars=10)
        with patch('llm.llmcli.common.prosody_chunker.time.perf_counter', return_value=0):
            self.assertEqual(list(chunker.feed(['He asked, "What is entropy?" Next'])), ['He asked, "What is entropy?"'])
            self.assertEqual(chunker.flush(), 'Next')

    def test_force_limit_does_not_separate_title_from_name(self):
        chunker = self.make_chunker(first_target_chars=100, max_chars=14, max_wait_s=0)
        with patch('llm.llmcli.common.prosody_chunker.time.perf_counter', return_value=1):
            self.assertEqual(list(chunker.feed(['Please ask Dr. '])), [])
            self.assertEqual(list(chunker.feed(['Taylor now '])), ['Please ask Dr. Taylor'])
            self.assertEqual(chunker.flush(), 'now')


if __name__ == '__main__':
    unittest.main()
