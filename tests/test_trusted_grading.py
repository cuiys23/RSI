import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
from utils.trusted_grading import evaluate


class TrustedGradingTests(unittest.TestCase):
    def test_sandbox_failure_keeps_partial_predictions(self):
        data = pd.DataFrame({'Grading ID': [str(i) for i in range(100)],
            'Points': ['6'] * 100, 'Problem': ['p']*100, 'Response': ['a']*100,
            'Solution': ['s']*100, 'Grading guidelines': ['g']*100})
        from utils.constants import REPO_NAME
        container = SimpleNamespace(attrs={'Mounts': [
            {'Destination': '/' + REPO_NAME, 'Source': 'safe'}], 'Config': {'Image': 'clean'}})
        with tempfile.TemporaryDirectory() as directory, \
                patch('utils.trusted_grading.pd.read_csv', return_value=data), \
                patch('utils.trusted_grading.run_candidate', side_effect=[
                    (6, None), ('None', 'Sandbox failure: worker stopped')]):
            with self.assertRaisesRegex(RuntimeError, 'worker stopped'):
                evaluate(container, directory, 'train', 'screen', 'fixed', 1)
            folder = Path(directory) / 'pre_eval'
            status = json.loads((folder / 'stage_status.json').read_text())
            self.assertEqual(status['completed_samples'], 2)
            self.assertFalse(status['success'])
            self.assertTrue((folder / 'predictions.csv').exists())
            self.assertIn('worker stopped', (folder / 'evaluation.jsonl').read_text())

    def test_labels_stay_host_side_and_screening_is_reused(self):
        calls = []
        def candidate(image, source, inputs, *args):
            self.assertEqual(set(inputs), {'domain', 'problem', 'student_answer', 'solution', 'grading_guidelines'})
            calls.append(inputs)
            return 6, None
        data = pd.DataFrame({'Grading ID': [str(i) for i in range(100)],
            'Points': ['6'] * 100, 'Problem': ['p']*100, 'Response': ['a']*100,
            'Solution': ['s']*100, 'Grading guidelines': ['g']*100})
        container = SimpleNamespace(attrs={'Mounts':[{'Destination':'/hyperagents', 'Source':'safe'}], 'Config':{'Image':'clean'}})
        from utils.constants import REPO_NAME
        container.attrs['Mounts'][0]['Destination'] = '/' + REPO_NAME
        real_read = pd.read_csv
        def read(path, **kwargs):
            if Path(path).name.startswith('gradingbench'):
                return data.copy()
            return real_read(path, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch('utils.trusted_grading.pd.read_csv', side_effect=read), patch('utils.trusted_grading.run_candidate', side_effect=candidate):
            evaluate(container, directory, 'train', 'screen', 'fixed', 1)
            self.assertEqual(len(calls), 10)
            evaluate(container, directory, 'train', 'full', 'fixed', 1)
            self.assertEqual(len(calls), 100)
            with self.assertRaises(ValueError):
                evaluate(container, directory, 'test', 'full', 'fixed', 1)
