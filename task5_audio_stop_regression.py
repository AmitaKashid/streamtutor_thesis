
"""Run from repository root: python task5_audio_stop_regression.py"""

from pathlib import Path
import ast
import unittest


class PlaybackProtocolRegression(unittest.TestCase):

    def test_stop_events_include_active_turn_id(self):
        s = Path('frontend_api.py').read_text(encoding='utf-8')

        ast.parse(s)

        self.assertIn(
            'active_audio_turn_id = payload.get("turn_id")', s
        )

        self.assertIn(
            '"turn_id": active_audio_turn_id', s
        )

        self.assertNotIn(
            'emit_from_thread({"type": "audio_stop"})', s
        )

        self.assertNotIn(
            'await ws_send({"type": "audio_stop"})', s
        )

    def test_browser_requires_matching_turn_id(self):
        s = Path('frontend/app/page.tsx').read_text(encoding='utf-8')

        self.assertIn(
            'data.turn_id === playbackTurnIdRef.current', s
        )

        self.assertIn(
            'stopBrowserAudio(data.turn_id)', s
        )

        self.assertIn('browser_audio_interrupted', s)

    def test_completion_requires_stream_end_and_drained_buffers(self):
        s = Path('frontend/app/page.tsx').read_text(encoding='utf-8')

        self.assertIn('!streamEndedRef.current', s)

        self.assertIn(
            'playbackQueueRef.current.length > 0', s
        )

        self.assertIn('currentSourceRef.current', s)


if __name__ == '__main__':
    unittest.main(verbosity=2)
