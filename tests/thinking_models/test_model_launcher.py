import unittest
from unittest.mock import patch

from scripts import start_all_models as launcher


class LauncherTests(unittest.TestCase):
    @patch.object(launcher, 'ssh', return_value='509510|RUNNING|gpu08')
    def test_existing_job_is_reused_without_submitting(self, ssh):
        self.assertEqual(launcher.ensure_job('llmtutor_bonsai2', 'bonsai2_9b_thinking', 18086, 'scripts/job.sbatch'), '509510')
        self.assertEqual(ssh.call_count, 1)
        self.assertIn('squeue', ssh.call_args.args[0])

    @patch.object(launcher, 'ssh', side_effect=['', '509511'])
    def test_missing_job_is_submitted(self, ssh):
        self.assertEqual(launcher.ensure_job('llmtutor_e4b', 'gemma4_e4b_thinking', 8005, 'scripts/job.sbatch'), '509511')
        command = ssh.call_args.args[0]
        self.assertIn('sbatch --parsable', command)
        self.assertIn('gemma4_e4b_thinking 8005', command)

    @patch.object(launcher, 'ssh', return_value='1|RUNNING|gpu01\n2|PENDING|(null)')
    def test_duplicate_jobs_are_not_silently_selected(self, ssh):
        with self.assertRaisesRegex(RuntimeError, 'Multiple'):
            launcher.jobs('llmtutor_e4b')

    @patch.object(launcher, 'jobs', return_value=[['509510', 'RUNNING', 'gpu08']])
    def test_running_job_returns_node(self, jobs):
        self.assertEqual(launcher.wait_node('llmtutor_bonsai2', '509510', 10), 'gpu08')

    @patch.object(launcher, 'jobs', return_value=[])
    def test_failed_job_reports_log_location(self, jobs):
        with self.assertRaisesRegex(RuntimeError, 'llmtutor_bonsai2_509510.log'):
            launcher.wait_node('llmtutor_bonsai2', '509510', 10)


if __name__ == '__main__':
    unittest.main()
