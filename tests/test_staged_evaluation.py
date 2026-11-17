import ast
import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


class StagedEvaluationTests(unittest.TestCase):
    def test_screen_full_and_failure(self):
        tree = ast.parse(Path('generate_loop.py').read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'eval_produced_agent')
        calls = []
        container = Mock()
        container.exec_run.side_effect = lambda cmd, **kw: (calls.append(cmd) or SimpleNamespace(exit_code=0, output=b''))
        def copy_output(container, source_path, dest_path):
            count = 10 if source_path.endswith('_screen') else 100
            with open(Path(dest_path) / 'predictions.csv', 'w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['question_id', 'prediction'])
                writer.writeheader()
                writer.writerows({'question_id': str(i), 'prediction': 'None'} for i in range(count))
            (Path(dest_path) / 'report.json').write_text(json.dumps({'total': count, 'total_correct': 1}))
        namespace = dict(TASK_MODEL='test', os=os, json=json, REPO_NAME='repo', log_container_output=Mock(), copy_from_container=copy_output)
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'generate_loop.py', 'exec'), namespace)
        evaluate = namespace['eval_produced_agent']
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate(container, '/tmp', directory, 'paper_review', stage='screen', eval_samples=10)
            self.assertEqual(result['total'], 10)
            harness = next(c for c in calls if 'domains.harness' in c)
            self.assertEqual(harness[harness.index('--num_samples')+1], '10')
            self.assertEqual(harness[harness.index('--subset')+1], '_filtered_100_train')
            calls.clear()
            evaluate(container, '/tmp', directory, 'paper_review', stage='full', eval_samples=100)
            self.assertTrue(any(c[:2] == ['cp', '-a'] for c in calls))
            self.assertTrue((Path(directory) / 'paper_review_eval_screen/report.json').exists())
            self.assertTrue((Path(directory) / 'paper_review_eval/report.json').exists())
            container.exec_run.side_effect = None
            container.exec_run.return_value = SimpleNamespace(exit_code=1, output=b'failure detail')
            with self.assertRaisesRegex(RuntimeError, 'failure detail'):
                evaluate(container, '/tmp', directory, 'paper_review', split='val')
            status = json.loads((Path(directory) / 'paper_review_eval_val/stage_status.json').read_text())
            self.assertFalse(status['success'])


if __name__ == '__main__':
    unittest.main()
