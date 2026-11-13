"""A stop received before a pipeline starts must not be cleared on entry."""
import threading
import types
import unittest
from unittest.mock import Mock
from test_pipeline_failure import load_pipeline


class StopBeforeStartTests(unittest.TestCase):
    def test_pending_stop_prevents_generation(self):
        run = load_pipeline()
        cancel = threading.Event(); cancel.set()
        playback = Mock(); playback.interrupted = threading.Event()
        llm = Mock(); events = []
        run(transcript='question', stt_time=0, total_timer=Mock(), llm=llm,
            tts_engine=None, retriever=None, args=types.SimpleNamespace(),
            playback=playback, pipeline_active=threading.Event(), judge=None,
            request_cancel_event=cancel, event_cb=events.append)
        self.assertTrue(playback.interrupted.is_set())
        llm.stream.assert_not_called()
        self.assertFalse(any(event.get('type') == 'tutor_turn_started' for event in events))

if __name__ == '__main__': unittest.main()
