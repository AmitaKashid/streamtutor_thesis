import unittest
from pathlib import Path
from task7_run_block_guarded import validate_live_block

class GuardTests(unittest.TestCase):
    def setUp(self):
        self.manifest={'schedule_sha256':'schedule','registry_sha256':'registry'}
        self.live=dict(block_id='b02_on',condition='on',schedule_sha256='schedule',registry_sha256='registry',telemetry_path=str(Path('b02_on.events.jsonl').resolve()))
    def test_matching_live_block_passes(self):
        validate_live_block(self.live,'b02_on','on',self.manifest,'b02_on.events.jsonl')
    def test_stale_off_backend_rejected(self):
        x=dict(self.live,block_id='b01_off',condition='off')
        with self.assertRaisesRegex(ValueError,'WRONG LIVE BACKEND'):
            validate_live_block(x,'b02_on','on',self.manifest,'b02_on.events.jsonl')
    def test_wrong_registry_rejected(self):
        x=dict(self.live,registry_sha256='different')
        with self.assertRaises(ValueError):
            validate_live_block(x,'b02_on','on',self.manifest,'b02_on.events.jsonl')
    def test_wrong_telemetry_rejected(self):
        x=dict(self.live,telemetry_path=str(Path('b01_off.events.jsonl').resolve()))
        with self.assertRaisesRegex(ValueError,'TELEMETRY'):
            validate_live_block(x,'b02_on','on',self.manifest,'b02_on.events.jsonl')

if __name__=='__main__':unittest.main()
