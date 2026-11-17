"""Test the grading method in isolation without API SDKs or credentials."""
import ast
import json
import unittest
from pathlib import Path
from unittest.mock import Mock


class MinimalGraderTests(unittest.TestCase):
    def run_grader(self, response):
        tree = ast.parse(Path('task_agent.py').read_text(encoding='utf-8'))
        method = next(n for n in tree.body if isinstance(n, ast.ClassDef)).body[-1]
        namespace = {'json': json, 'get_response_from_llm': Mock(
            return_value=(response, [{'text': response}], {}))}
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'task_agent.py', 'exec'), namespace)
        agent = Mock(model='test-model')
        inputs = dict(domain='imo_grading', problem='problem', student_answer='answer', solution='reference', grading_guidelines='rubric')
        prediction, history = namespace['forward'](agent, inputs)
        namespace['get_response_from_llm'].assert_called_once()
        self.assertEqual(history, [{'text': response}])
        return prediction, agent

    def test_allowed_integer_scores(self):
        for score in (0, 1, 6, 7):
            with self.subTest(score=score):
                prediction, _ = self.run_grader(json.dumps({'response': score}))
                self.assertEqual(prediction, score)
                self.assertIs(type(prediction), int)

    def test_invalid_outputs_fail_without_retry(self):
        for response in ('bad json', '{}', '[]', '{"response": "6"}',
                         '{"response": 6.0}', '{"response": true}',
                         '{"response": 2}', '{"response": null}',
                         '{"response": 7, "explanation": "ok"}'):
            with self.subTest(response=response):
                prediction, agent = self.run_grader(response)
                self.assertEqual(prediction, 'None')
                self.assertTrue(agent.log.call_args.args[0].startswith('Grading failed:'))


if __name__ == '__main__':
    unittest.main()
