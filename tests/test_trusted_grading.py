import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
from utils.trusted_grading import evaluate


class TrustedGradingTests(unittest.TestCase):
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
