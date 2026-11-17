import ast
import csv
import json
import math
import os
import tempfile
import unittest
from pathlib import Path


class ValidationSelectionTests(unittest.TestCase):
    def test_complete_validation_only(self):
        tree = ast.parse(Path('utils/gl_utils.py').read_text(encoding='utf-8'))
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in {'get_full_validation_score', 'is_starting_node'}]
        namespace = dict(os=os, json=json, math=math)
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'gl_utils.py', 'exec'), namespace)
        score = namespace['get_full_validation_score']
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'gen_initial/imo_grading_eval_val'
            folder.mkdir(parents=True)
            with open(folder / 'predictions.csv', 'w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['Grading ID', 'prediction'])
                writer.writerows((i, 6) for i in range(100))
            report = dict(total=100, total_correct=80, overall_accuracy=0.8)
            (folder / 'report.json').write_text(json.dumps(report))
            self.assertEqual(score('imo_grading', directory, 'initial'), 0.8)
            report['total'] = 10
            (folder / 'report.json').write_text(json.dumps(report))
            self.assertIsNone(score('imo_grading', directory, 'initial'))
            report['total'] = 100
            (folder / 'report.json').write_text(json.dumps(report))
            (folder / 'stage_status.json').write_text(json.dumps({'success': False}))
            self.assertIsNone(score('imo_grading', directory, 'initial'))
            self.assertIsNone(score('imo_grading', directory, 1))


if __name__ == '__main__':
    unittest.main()
