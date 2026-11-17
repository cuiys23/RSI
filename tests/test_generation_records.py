import json
import tempfile
import unittest
from pathlib import Path

from utils.generation_records import initialize_record, finalize_record, evaluation_directory


class GenerationRecordTests(unittest.TestCase):
    def test_failed_attempt_retains_patch_parent_and_stage_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = dict(gen_output_dir=directory, current_genid=3, parent_genid=1,
                            prev_patch_files=['parent.diff'], curr_patch_files=[],
                            parent_agent_success=False, evaluation_error='Process exit 1')
            initialize_record(metadata, ['imo_grading'], 'meta', 'task')
            raw = Path(directory) / 'agent_output'
            raw.mkdir()
            (raw / 'model_patch.diff').write_text('partial change', encoding='utf-8')
            (raw / 'meta_agent_chat_history.md').write_text('model trace', encoding='utf-8')
            finalize_record(metadata, ['imo_grading'])
            saved = json.loads((Path(directory) / 'metadata.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['parent_genid'], 1)
            self.assertEqual(saved['prev_patch_files'], ['parent.diff'])
            self.assertEqual(saved['status'], 'failed')
            self.assertEqual(saved['failure_reason'], 'Process exit 1')
            self.assertEqual((Path(directory) / 'model_patch.diff').read_text(), 'partial change')
            self.assertEqual((Path(directory) / 'meta_agent_chat.md').read_text(), 'model trace')
            for stage in ('pre_eval', 'train_eval', 'val_eval'):
                status = json.loads((Path(directory) / stage / 'stage_status.json').read_text())
                self.assertEqual(status['status'], 'not_run')
                self.assertEqual(status['reason'], 'Process exit 1')

    def test_rejected_and_empty_attempts_keep_empty_diff(self):
        for fields, expected in ((dict(screening_rejected=True), 'screening_rejected'),
                                 (dict(no_changes=True), 'no_changes')):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                metadata = dict(gen_output_dir=directory, parent_agent_success=True, **fields)
                initialize_record(metadata, ['imo_grading'], 'meta', 'task')
                finalize_record(metadata, ['imo_grading'])
                self.assertEqual(metadata['status'], expected)
                self.assertEqual((Path(directory) / 'model_patch.diff').read_bytes(), b'')

    def test_legacy_directory_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / 'imo_grading_eval_val'
            old.mkdir()
            self.assertEqual(evaluation_directory(directory, 'imo_grading', 'val', existing=True), old)
            canonical = Path(directory) / 'val_eval'
            canonical.mkdir()
            (canonical / 'report.json').write_text('{}')
            self.assertEqual(evaluation_directory(directory, 'imo_grading', 'val', existing=True), canonical)


if __name__ == '__main__':
    unittest.main()
