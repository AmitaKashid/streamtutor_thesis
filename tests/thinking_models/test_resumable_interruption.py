"""Exercise the production interruption decision without loading GPU models."""
import ast
import re
import threading
import types
import unittest

from test_voice_latency import ROOT, load_method_nodes


def interruption_fixture():
    events, calls = [], []
    busy = threading.Event(); busy.set()
    microphone = threading.Event(); microphone.set()
    browser_active = threading.Event(); browser_active.set()
    env = {
        'interrupt_capture_mode': True, 'interrupt_capture_turn_id': 'old',
        'interrupt_context_at_pause': ['prior sentence'],
        'interrupt_capture_started_at': 1., 'interrupt_last_voice_at': 2.,
        'active_audio_turn_id': 'old', 'session_closed': threading.Event(),
        'mic_enabled': microphone, 'tts_busy': busy, 'browser_audio_active': browser_active,
        'playback': types.SimpleNamespace(
            get_interrupt_snapshot=lambda: ['prior sentence'],
            get_interrupt_context=lambda **kwargs: [],
            stop_now=lambda: calls.append('stop')),
        'interrupt_segmenter': types.SimpleNamespace(reset=lambda: calls.append('reset_segmenter')),
        'reset_partial_state': lambda: calls.append('reset_partial'),
        'reset_tts_probe_state': lambda: calls.append('reset_probe'),
        'drain_after_tts': types.SimpleNamespace(clear=lambda: calls.append('clear_drain')),
        'cancel_all_browser_delivery': lambda: calls.append('cancel_delivery'),
        'clear_frontend_transcript': lambda: calls.append('clear_transcript'),
        'emit_from_thread': events.append,
        'launch_pipeline': lambda **kwargs: calls.append(('launch', kwargs)),
        'is_unusable_stt_text': lambda text: not text or 'watching' in text,
        'is_valid_barge_in_text': lambda text, last: text != 'echo',
    }
    load_method_nodes(ROOT / 'frontend_api.py', ['complete_interruption'], env)
    return env, events, calls


class ResumableInterruptionTests(unittest.TestCase):
    def test_unrecognized_noise_and_echo_resume_without_cancelling(self):
        for transcript in ('', 'echo', 'thanks for watching'):
            with self.subTest(transcript=transcript):
                env, events, calls = interruption_fixture()
                self.assertTrue(env['complete_interruption']('old', transcript, .25))
                self.assertEqual(events, [{'type': 'audio_resume', 'turn_id': 'old'},
                                          {'type': 'stage', 'value': 'speaking'}])
                self.assertNotIn('stop', calls)
                self.assertNotIn('cancel_delivery', calls)
                self.assertFalse(any(isinstance(item, tuple) and item[0] == 'launch' for item in calls))

    def test_valid_interruption_stops_the_old_turn_and_submits_the_new_question(self):
        env, events, calls = interruption_fixture()
        self.assertTrue(env['complete_interruption']('old', 'Explain entropy', .4))
        self.assertEqual([event['type'] for event in events],
                         ['audio_stop', 'barge_in_accepted', 'stage'])
        self.assertIn('stop', calls); self.assertIn('cancel_delivery', calls)
        launched = next(item[1] for item in calls if isinstance(item, tuple) and item[0] == 'launch')
        self.assertEqual(launched['transcript'], 'Explain entropy')
        self.assertEqual(launched['interrupted_chunks'], ['prior sentence'])

    def test_stale_recognition_cannot_cancel_a_new_turn_or_restart_stopped_audio(self):
        for change in ('new_turn', 'mic_off', 'stopped'):
            with self.subTest(change=change):
                env, events, calls = interruption_fixture()
                if change == 'new_turn': env['active_audio_turn_id'] = 'new'
                if change == 'mic_off': env['mic_enabled'].clear()
                if change == 'stopped': env['interrupt_capture_mode'] = False
                self.assertFalse(env['complete_interruption']('old', 'Explain entropy', .4))
                self.assertEqual(events, []); self.assertEqual(calls, [])

    def test_legitimate_words_are_not_blocked_by_a_phrase_list(self):
        env = {'re': re}
        load_method_nodes(ROOT / 'frontend_api.py', ['normalize_stt_text', 'is_unusable_stt_text'], env)
        self.assertFalse(env['is_unusable_stt_text']('Thank you. Bye.'))
        self.assertFalse(env['is_unusable_stt_text']('Explain entropy and decision trees.'))
        self.assertTrue(env['is_unusable_stt_text']('...'))


if __name__ == '__main__':
    unittest.main()
