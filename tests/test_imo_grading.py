import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from domains.imo.curate_subsets import curate_subsets_gradingbench
from domains.imo.grading_utils import normalize_points
from domains.report import report


class GradingEvaluationTests(unittest.TestCase):
    def test_invalid_predictions_remain_in_denominator(self):
        predictions = ['0', ' 1.0 ', '6', '7', '', None, 'correct', '2', '6.5', 'NaN']
        with tempfile.TemporaryDirectory() as directory:
            pd.DataFrame({
                'Grading ID': [str(i) for i in range(10)],
                'Points': [0, 1, 6, 7, 0, 1, 7, 0, 6, 0],
                'Reward': ['ignored'] * 10,
                'prediction': predictions,
            }).to_csv(Path(directory) / 'predictions.csv', index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                result, _ = report(directory, 'imo_grading')
            self.assertEqual(result['total'], 10)
            self.assertEqual(result['total_correct'], 4)
            self.assertEqual(result['overall_accuracy'], 0.4)
            self.assertEqual(len(result['question_ids_failed']), 6)

    def test_normalization(self):
        for value in [None, True, '', 'None', 'incorrect', '7.1', 'inf', 'NaN', 2]:
            self.assertIsNone(normalize_points(value))
        self.assertEqual(normalize_points(' 6.0 '), 6)

    def test_manifest_reuses_disjoint_stratified_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'gradingbench.csv'
            frame = pd.DataFrame({
                'Grading ID': [str(i) for i in range(400)],
                'Points': [0, 1, 6, 7] * 100,
            })
            frame.to_csv(path, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                curate_subsets_gradingbench(path)
            manifest_path = Path(directory) / 'grading_split_ids.json'
            before = manifest_path.read_bytes()
            splits = json.loads(before)['splits']
            self.assertEqual(len(set(sum(splits.values(), []))), 300)
            for name in splits:
                part = pd.read_csv(Path(directory) / f'gradingbench_filtered_100_{name}.csv')
                self.assertEqual(len(part), 100)
                self.assertEqual(part['Points'].value_counts().to_dict(), {0:25, 1:25, 6:25, 7:25})
            frame.sample(frac=1, random_state=9).to_csv(path, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                curate_subsets_gradingbench(path)
            self.assertEqual(before, manifest_path.read_bytes())
            for name, ids in splits.items():
                part = pd.read_csv(Path(directory) / f'gradingbench_filtered_100_{name}.csv', dtype=str)
                self.assertEqual(part['Grading ID'].tolist(), ids)


if __name__ == '__main__':
    unittest.main()
