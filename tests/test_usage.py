import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from utils.experiment import record_usage, summarize_usage


class UsageTests(unittest.TestCase):
    def test_categories_and_missing_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'usage.jsonl')
            with patch.dict(os.environ, RSI_USAGE_PATH=path, RSI_GENERATION='3', RSI_CATEGORY='self_modification'):
                record_usage(dict(requested_model='fixed', actual_model='actual', input_tokens=10, output_tokens=4, status='success'))
                record_usage(dict(input_tokens=None, output_tokens=None, status='error'))
            result = summarize_usage(path)['self_modification']
            self.assertEqual(result['input_tokens'], 10)
            self.assertEqual(result['output_tokens'], 4)
            self.assertEqual(result['missing_usage'], 1)
            self.assertEqual(result['failed_requests'], 1)
            first = json.loads(Path(path).read_text().splitlines()[0])
            self.assertEqual(first['generation'], 3)
            self.assertEqual(first['actual_model'], 'actual')
