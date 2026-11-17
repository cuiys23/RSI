"""Exercise record persistence through the actual generation lifecycle."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from generate_loop import generate


class GenerationFailureTests(unittest.TestCase):
    def run_generation(self, directory, container, **kwargs):
        with patch('generate_loop.build_container', return_value=container), \
                patch('generate_loop.cleanup_container') as cleanup, \
                patch('generate_loop.apply_diffs_container', return_value='parent-commit'), \
                patch('generate_loop.copy_from_container'), \
                patch('generate_loop.run_commands_to_check_compilation'), \
                patch('generate_loop.get_full_validation_score', return_value=None), \
                patch('generate_loop.get_score', return_value=None):
            cleanup.side_effect = RuntimeError('cleanup failed') if container is not None else None
            result = generate(None, ['imo_grading'], directory, 'test', 1, 0,
                              directory, max_generation=30, eval_samples=[100],
                              eval_subsets=['_filtered_100_train'], **kwargs)
        return result

    def test_container_creation_failure_still_creates_full_record(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_generation(directory, None)
            root = Path(directory) / 'gen_1'
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['parent_genid'], 0)
            self.assertIn('Container creation failed', result['failure_reason'])
            self.assertTrue((root / 'model_patch.diff').exists())
            self.assertTrue((root / 'meta_agent_chat.md').exists())
            self.assertTrue((root / 'failure.log').exists())
            self.assertEqual(json.loads((root / 'metadata.json').read_text())['status'], 'failed')

    def test_cleanup_failure_does_not_overwrite_screening_rejection(self):
        container = Mock()
        container.exec_run.return_value = SimpleNamespace(exit_code=0, output=b'')
        with tempfile.TemporaryDirectory() as directory, \
                patch('generate_loop.eval_produced_agent', return_value={'total_correct': 0}):
            result = self.run_generation(directory, container, run_meta_agent=False)
            self.assertEqual(result['status'], 'screening_rejected')
            self.assertEqual(result['cleanup_errors'], ['cleanup failed'])
            self.assertFalse(result['valid_parent'])
            self.assertEqual(result['parent_genid'], 0)

    def test_meta_command_and_exports_use_linux_container_paths(self):
        container = Mock()
        container.exec_run.return_value = SimpleNamespace(exit_code=0, output=b'')
        with tempfile.TemporaryDirectory() as directory, \
                patch('generate_loop.copy_prev_eval_to_container', return_value='/tmp/feedback'):
            self.run_generation(directory, container)
        command = next(call.kwargs['cmd'] for call in container.exec_run.call_args_list
                       if isinstance(call.kwargs.get('cmd'), list)
                       and 'run_meta_agent.py' in call.kwargs['cmd'])
        self.assertEqual(command[command.index('--chat_history_file') + 1],
                         '/tmp/agent_output/meta_agent_chat_history.md')
        self.assertEqual(command[command.index('--outdir') + 1], '/tmp/agent_output')
        self.assertEqual(command[command.index('--evals_folder') + 1], '/tmp/feedback')


if __name__ == '__main__':
    unittest.main()
