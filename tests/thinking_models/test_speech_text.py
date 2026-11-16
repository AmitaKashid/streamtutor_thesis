import unittest
from llm.llmcli.common.speech_text import speech_text


class SpeechTextTests(unittest.TestCase):
    def test_speech_removes_basic_formatting(self):
        self.assertEqual(speech_text('### How it works\n- **Root:** begin with all data.'),
                         'How it works Root: begin with all data.')
        self.assertEqual(speech_text('A `tree` uses [examples](https://example.com).'),
                         'A tree uses examples.')

    def test_chunk_split_delimiters_are_not_spoken(self):
        self.assertEqual(speech_text('**Decision'), 'Decision')
        self.assertEqual(speech_text('tree** is useful.'), 'tree is useful.')

    def test_regular_content_is_preserved(self):
        self.assertEqual(speech_text('C# and model_name cost 2 * 3.'), 'C# and model_name cost 2 times 3.')

if __name__ == '__main__':
    unittest.main()
