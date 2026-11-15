import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
from app.word_alignment import WordAlignmentError, align_with_whisper, bind_alignment_to_text


class WordAlignmentTests(unittest.TestCase):
    def test_preserves_words_whitespace_and_punctuation(self):
        text = 'Hello,\nworld!'
        alignment = [
            {'word': 'Hello', 'start': .1, 'end': .4},
            {'word': ',', 'start': .4, 'end': .4},
            {'word': ' world', 'start': .7, 'end': 1.1},
            {'word': '!', 'start': 1.1, 'end': 1.1},
        ]
        result = bind_alignment_to_text(text, alignment, 1.5)
        self.assertEqual([item['text'] for item in result], ['Hello,', '\nworld!'])
        self.assertEqual([item['start'] for item in result], [.1, .7])
        self.assertEqual(''.join(item['text'] for item in result), text)

    def test_rejects_missing_or_invalid_timings_without_inventing_words(self):
        cases = [
            [{'word': 'Hello', 'start': 0, 'end': .2}],
            [{'word': 'Hello', 'start': .5, 'end': .2}, {'word': ' world', 'start': .6, 'end': 1}],
            [{'word': 'Hello', 'start': 0, 'end': float('nan')}, {'word': ' world', 'start': .6, 'end': 1}],
            [{'word': 'Other', 'start': 0, 'end': .2}, {'word': ' world', 'start': .6, 'end': 1}],
        ]
        for alignment in cases:
            with self.subTest(alignment=alignment), self.assertRaises(WordAlignmentError):
                bind_alignment_to_text('Hello world', alignment, 2.0)

    def test_faster_whisper_alignment_uses_known_text_and_resampled_audio(self):
        observed = {}
        class Tokenizer:
            def __init__(self, tokenizer, multilingual, **kwargs):
                observed['language'] = kwargs['language']
            def encode(self, text):
                observed['text'] = text
                return [1, 2]
        class Extractor:
            nb_max_frames = 3000
            def __call__(self, samples):
                observed['samples'] = len(samples)
                return np.zeros((80, 101), dtype=np.float32)
        class Model:
            feature_extractor = Extractor()
            hf_tokenizer = object()
            model = types.SimpleNamespace(is_multilingual=True)
            max_length = 448
            def encode(self, features):
                observed['features_shape'] = features.shape
                return 'encoded'
            def find_alignment(self, tokenizer, tokens, encoded, frames):
                observed['frames'] = frames
                self.assertions = (tokens, encoded)
                return [[{'word': 'Hello', 'start': .1, 'end': .4},
                         {'word': ' world', 'start': .7, 'end': 1.1}]]
        audio_module = types.ModuleType('faster_whisper.audio')
        audio_module.pad_or_trim = lambda features: features
        tokenizer_module = types.ModuleType('faster_whisper.tokenizer')
        tokenizer_module.Tokenizer = Tokenizer
        parent_module = types.ModuleType('faster_whisper')
        with patch.dict(sys.modules, {'faster_whisper': parent_module,
                                      'faster_whisper.audio': audio_module,
                                      'faster_whisper.tokenizer': tokenizer_module}):
            words = align_with_whisper(Model(), 'Hello world ', np.ones((48000, 1)), 24000)
        self.assertEqual(observed['samples'], 32000)
        self.assertEqual(observed['text'], 'Hello world')
        self.assertEqual(observed['frames'], 100)
        self.assertEqual(''.join(word['text'] for word in words), 'Hello world ')
        self.assertEqual([word['start'] for word in words], [.1, .7])


if __name__ == '__main__':
    unittest.main()
