import unittest
from task7_run_block import terminal

class Task7ExecutionTests(unittest.TestCase):
    def test_ack_requires_audio_done(self):
        e=[dict(event='tutor_turn_started',session_id='s',turn_id='t',mono_ns=1),dict(event='browser_audio_finished',session_id='s',turn_id='t',mono_ns=8)]
        self.assertIsNone(terminal(e,set()))
    def test_ack_after_done(self):
        e=[dict(event='tutor_turn_started',session_id='s',turn_id='t',mono_ns=1),dict(event='audio_done',session_id='s',turn_id='t',mono_ns=3),dict(event='browser_audio_finished',session_id='s',turn_id='t',mono_ns=8)]
        self.assertEqual(terminal(e,set())[0],'passed')
    def test_duplicate_ack_rejected(self):
        e=[dict(event='tutor_turn_started',session_id='s',turn_id='t',mono_ns=1),dict(event='audio_done',session_id='s',turn_id='t',mono_ns=3),dict(event='browser_audio_finished',session_id='s',turn_id='t',mono_ns=8),dict(event='browser_audio_finished',session_id='s',turn_id='t',mono_ns=9)]
        self.assertEqual(terminal(e,set())[0],'failed')
    def test_old_turn_excluded(self):
        e=[dict(event='tutor_turn_started',session_id='s',turn_id='t',mono_ns=1),dict(event='audio_done',session_id='s',turn_id='t',mono_ns=3),dict(event='browser_audio_finished',session_id='s',turn_id='t',mono_ns=8)]
        self.assertIsNone(terminal(e,{('s','t')}))
if __name__=='__main__':unittest.main()
