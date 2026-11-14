"""Session reset preserves an active or queued response and clears omission state."""
import threading
import unittest
from app.context_reset import clear_context_if_idle
from test_context_measurements import load_memory_class


class ClearContextTest(unittest.TestCase):
    def setUp(self):
        self.memory = load_memory_class()(max_turns=1)
        self.memory.add_turn('old', 'answer')
        self.memory.add_turn('new', 'answer')
        self.pipeline = threading.Lock()
        self.request = threading.Lock()
        self.pending = set()
        self.mic = threading.Event()
        self.audio_processing = threading.Event()

    def clear(self):
        return clear_context_if_idle(self.memory, self.pipeline, self.request, self.pending, self.mic, self.audio_processing)

    def test_idle_clears_turns_and_omission_marker(self):
        self.assertTrue(self.clear())
        self.assertEqual(self.memory.snapshot_with_omission(), ([], False))
        self.memory.add_turn('fresh', 'answer')
        self.assertEqual(len(self.memory.snapshot()), 1)

    def test_queued_or_running_turn_cannot_repopulate_after_clear(self):
        self.pending.add(threading.Event())
        self.assertFalse(self.clear())
        self.pending.clear()
        with self.pipeline:
            self.assertFalse(self.clear())
        self.assertEqual(len(self.memory.snapshot()), 1)

    def test_inflight_audio_processing_blocks_clear(self):
        self.audio_processing.set()
        self.assertFalse(self.clear())
        self.assertEqual(len(self.memory.snapshot()), 1)

    def test_live_microphone_must_be_stopped(self):
        self.mic.set()
        self.assertFalse(self.clear())
        self.assertEqual(len(self.memory.snapshot()), 1)

if __name__ == '__main__': unittest.main()
