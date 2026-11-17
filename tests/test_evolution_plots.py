import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
from analysis.plot_progress import plot_progress_single
from analysis.visualize_archive import build_graph_single_domain, visualize_graph


class EvolutionPlotTests(unittest.TestCase):
    def make_run(self, root):
        archive = []
        entries = []
        for node, state, parent, score in (
            (0, 'completed', None, 0.0),
            (1, 'failed', 0, None),
            (2, 'screening_rejected', 0, None),
            (3, 'completed', 0, 0.8),
            (4, 'completed', 3, 0.6),
            (5, 'no_changes', None, None),
        ):
            folder = root / f'gen_{node}'
            folder.mkdir()
            (folder / 'metadata.json').write_text(json.dumps(dict(
                parent_genid=parent, status=state, run_full_eval=score is not None,
                parent_agent_success=state != 'failed', valid_parent=score is not None,
                screening_rejected=state == 'screening_rejected', curr_patch_files=[])))
            if score is not None:
                stage = folder / 'val_eval'
                stage.mkdir()
                (stage / 'stage_status.json').write_text(json.dumps(dict(
                    stage='full', split='val', success=True)))
                (stage / 'report.json').write_text(json.dumps(dict(
                    total=100, total_correct=int(score * 100), overall_accuracy=score)))
                with (stage / 'predictions.csv').open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['Grading ID', 'prediction'])
                    writer.writerows((i, 6) for i in range(100))
            archive.append(node)
            entries.append(dict(archive=list(archive)))
        (root / 'archive.jsonl').write_text('\n'.join(json.dumps(e) for e in entries) + '\n')
        return entries

    def test_all_nodes_states_and_original_ids_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = self.make_run(root)
            graph, scores = build_graph_single_domain('imo_grading', entries, directory, split='val')
            self.assertEqual(set(graph.nodes), {'0', '1', '2', '3', '4', '5'})
            self.assertEqual(set(graph.edges), {('0','1'), ('0','2'), ('0','3'), ('3','4')})
            self.assertEqual(graph.nodes['1']['status'], 'failed')
            self.assertEqual(graph.nodes['2']['status'], 'screening_rejected')
            self.assertEqual(scores['0'], 0.0)
            self.assertIsNone(scores['2'])
            # Exercise rendering without Graphviz, including an isolated node.
            with patch('analysis.visualize_archive.graphviz_layout', side_effect=ImportError()):
                visualize_graph(graph, scores, directory, 'imo_grading', split='val', save_svg=True)
            payload = json.loads((root / 'archive_graph_imo_grading_val_agent.json').read_text())
            self.assertEqual(len(payload['nodes']), 6)
            self.assertTrue((root / 'archive_graph_imo_grading_val_agent.svg').exists())
            short, _ = build_graph_single_domain('imo_grading', entries, directory, trunc_its=2, split='val')
            self.assertEqual(set(short.nodes), {'0', '1'})

    def test_curve_preserves_gaps_and_real_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_run(root)
            plot_progress_single('imo_grading', directory, split='val', svg=True)
            data = json.loads((root / 'progress_data_imo_grading_val_agent.json').read_text())
            self.assertEqual([entry['score'] for entry in data], [0.0, None, None, 0.8, 0.6, None])
            self.assertEqual([entry['best_score'] for entry in data], [0.0, 0.0, 0.0, 0.8, 0.8, 0.8])
            self.assertEqual([entry['iteration'] for entry in data], list(range(6)))


if __name__ == '__main__':
    unittest.main()
