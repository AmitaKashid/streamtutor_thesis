import unittest
from unittest.mock import patch

from scripts import start_all_models as launcher


class LauncherTests(unittest.TestCase):
    def test_summary_lists_all_models_and_judge_with_observed_status(self):
        from contextlib import redirect_stdout
        from io import StringIO
        output = StringIO()
        with patch.object(launcher, 'jobs', return_value=[['123', 'RUNNING', 'gpu07']]), \
             patch.object(launcher, 'healthy', return_value=True), redirect_stdout(output):
            launcher.print_service_summary()
        text = output.getvalue()
        for label, name, port in launcher.SERVICES:
            self.assertIn(label, text)
            self.assertIn(f'localhost:{port} healthy', text)
        self.assertIn('does not verify model identity', text)
        self.assertEqual(len(launcher.SERVICES), 8)

    def test_model_identity_accepts_exact_deployed_path(self):
        payload = {'data': [{'id': '/home/user/models/gemma-4-E4B_q4_0-it.gguf'}]}
        self.assertEqual(launcher.verify_model_identity('gemma4_e4b_thinking', payload), 'gemma-4-E4B_q4_0-it.gguf')

    def test_model_identity_rejects_other_model_and_malformed_payload(self):
        with self.assertRaisesRegex(RuntimeError, 'server reports'):
            launcher.verify_model_identity('gemma4_e4b_thinking', {'data': [{'id': '/models/Ternary-Bonsai-2-9B-Q2_0.gguf'}]})
        with self.assertRaisesRegex(RuntimeError, 'invalid'):
            launcher.verify_model_identity('gemma4_e4b_thinking', {'status': 'ok'})

    @patch.object(launcher, 'remote_models', return_value={'data': [{'id': '/models/Ternary-Bonsai-2-9B-Q2_0.gguf'}]})
    @patch.object(launcher, 'jobs', return_value=[['509510', 'RUNNING', 'gpu08']])
    def test_remote_identity_checks_assigned_node_and_configured_port(self, jobs, remote):
        launcher.wait_remote_model('llmtutor_bonsai2', '509510', 'gpu08', 18086, 'bonsai2_9b_thinking', timeout=1)
        remote.assert_called_once_with('gpu08', 18086)

    @patch.object(launcher, 'remote_models', return_value={'data': [{'id': '/models/unrelated.gguf'}]})
    @patch.object(launcher, 'jobs', return_value=[['509510', 'RUNNING', 'gpu08']])
    def test_wrong_server_on_remote_port_fails_instead_of_reusing(self, jobs, remote):
        with self.assertRaisesRegex(RuntimeError, 'server reports'):
            launcher.wait_remote_model('llmtutor_bonsai2', '509510', 'gpu08', 18086, 'bonsai2_9b_thinking', timeout=1)

    @patch.object(launcher, 'remote_models')
    @patch.object(launcher, 'jobs', return_value=[['509510', 'RUNNING', 'gpu09']])
    def test_moved_job_does_not_probe_stale_node(self, jobs, remote):
        with self.assertRaisesRegex(RuntimeError, 'no longer running'):
            launcher.wait_remote_model('llmtutor_bonsai2', '509510', 'gpu08', 18086, 'bonsai2_9b_thinking', timeout=1)
        remote.assert_not_called()

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
