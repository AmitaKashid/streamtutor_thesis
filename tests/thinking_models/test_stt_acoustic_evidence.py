"""Check acoustic and model-evidence gates without loading speech models."""
import ast
import copy
import types
import unittest
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from stt.stt_types import TranscriptResult, Segment, Word

ROOT = Path(__file__).resolve().parents[2]


def load_class(path, name, env, bases=True):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    nodes = [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.ClassDef)
             and (n.name == name or (name == 'VADSegmenter' and n.name == 'Utterance'))]
    if not bases:
        for node in nodes: node.bases = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<evidence>', 'exec'), env)
    return env[name]


class AcousticEvidenceTests(unittest.TestCase):
    def test_vad_counts_speech_frames_and_resets_without_counting_silence(self):
        class Vad:
            def __init__(self, aggressiveness): pass
            def is_speech(self, frame, rate): return bool(frame[0])
        env = {'dataclass': dataclass, 'np': np, 'deque': deque, 'webrtcvad': types.SimpleNamespace(Vad=Vad),
               'TARGET_SR': 16000, 'FRAME_MS': 20, 'MAX_SILENCE_MS': 100, 'MAX_UTTERANCE_MS': 30000,
               'PRE_ROLL_MS': 250, '__name__': __name__}
        cls = load_class('stt/vad_segmenter.py', 'VADSegmenter', env)
        vad = cls()
        voice, quiet = bytes([1]) * 640, bytes(640)
        for i in range(6): self.assertIsNone(vad.process_frame(voice, i * .02))
        self.assertEqual(vad.voiced_ms, 120)
        for i in range(4): self.assertIsNone(vad.process_frame(quiet, .12 + i * .02))
        utt = vad.process_frame(quiet, .20)
        self.assertEqual(utt.voiced_ms, 120)
        self.assertEqual(vad.voiced_ms, 0)

    def test_whisper_rejects_low_confidence_silence_without_phrase_matching(self):
        env = {'np': np, 'TranscriptResult': TranscriptResult, 'Segment': Segment, 'Word': Word}
        cls = load_class('stt/stt_backends/faster_whisper_backend.py', 'FasterWhisperBackend', env, bases=False)
        class Model:
            def transcribe(self, audio, **kwargs):
                return iter([
                    types.SimpleNamespace(text='Unrelated hallucination.', start=0, end=.5, no_speech_prob=.9, avg_logprob=-.9),
                    types.SimpleNamespace(text=' Thank you.', start=.5, end=1, no_speech_prob=.05, avg_logprob=-.1),
                ]), types.SimpleNamespace(language='en')
        backend = cls.__new__(cls); backend.model = Model(); backend.beam_size = 5; backend.engine = 'faster-whisper'
        audio = np.zeros(16000, dtype=np.float32)
        self.assertEqual(backend.transcribe_partial(audio, 16000, 'en').text, 'Thank you.')
        self.assertEqual(backend.transcribe(audio, 16000, 'en').text, 'Thank you.')


if __name__ == '__main__':
    unittest.main()
