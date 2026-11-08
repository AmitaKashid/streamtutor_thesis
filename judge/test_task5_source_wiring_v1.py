"""Static wiring checks for v3 browser acknowledgement protocol."""
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1]
class WiringTests(unittest.TestCase):
    def test_engine_exposes_candidate_sink(self):
        s=(ROOT/'app'/'mic_stt_llm_tts.py').read_text(encoding='utf-8')
        self.assertIn('evaluation_sink=None',s)
        self.assertIn('offer_evaluation_candidate(',s)
        self.assertIn('"type": "tutor_turn_started"',s)
        self.assertIn('"turn_id": evaluation_turn_id',s)
    def test_server_requires_turn_id_for_ack(self):
        s=(ROOT/'frontend_api.py').read_text(encoding='utf-8')
        self.assertIn('turn_id = data.get("turn_id")',s)
        self.assertIn('resolve_browser_delivery(',s)
        self.assertIn('browser_audio_interrupted',s)
        self.assertNotIn('browser_playback_ack',s)
    def test_client_waits_for_terminal_audio(self):
        s=(ROOT/'frontend'/'app'/'page.tsx').read_text(encoding='utf-8')
        self.assertIn('streamEndedRef.current = true',s)
        self.assertIn('maybeAcknowledgePlayback()',s)
        self.assertIn('browser_audio_interrupted',s)
        self.assertIn('notifiedTurnIdsRef.current.has(turnId)',s)
    def test_legacy_json_default_disabled(self):
        s=(ROOT/'frontend_api.py').read_text(encoding='utf-8')
        self.assertIn('os.environ.get("LLMTUTOR_RUNTIME_JUDGE", "0") == "1"',s)
if __name__=='__main__': unittest.main()
