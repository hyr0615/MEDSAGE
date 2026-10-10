"""CPU-only syntax, schema and reward checks using synthetic fixtures."""
import ast
import importlib.util
import json
import re
import sys
import unittest
from typing import Any
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures'
sys.path.insert(0, str(ROOT / 'rewards'))
from combined_reward import compute_score
from format_reward import format_reward


def response(box=None, answer='yes'):
    box = box or [0, 0, 10, 10]
    return ('<LOC>' + json.dumps({'bbox_xyxy_pixel': [box]}) + '</LOC>'
            '<VIS>Synthetic visual fixture.</VIS><KNO>Synthetic relation.</KNO>'
            '<CON>' + answer + '</CON>')


def ground_truth():
    return json.dumps({'answer': 'yes', 'task_type': 'full_LVKC', 'full_LVKC': True,
                       'reference_boxes': [[0, 0, 10, 10]],
                       'stage_availability': {'L': True, 'V': True, 'K': True, 'C': True}})


class ReleaseTests(unittest.TestCase):
    def test_public_examples_only_standard_fields(self):
        for row in json.loads((FIXTURES / 'sharegpt_sft.json').read_text()):
            self.assertEqual(set(row), {'conversations', 'images'})
            self.assertEqual([m['from'] for m in row['conversations']], ['human', 'gpt'])
            self.assertEqual(row['conversations'][0]['value'].count('<image>'), len(row['images']))
        for row in json.loads((FIXTURES / 'sharegpt_dpo.json').read_text()):
            self.assertEqual(set(row), {'conversations', 'images', 'chosen', 'rejected'})
            self.assertEqual(row['conversations'][-1]['from'], 'human')
            for name in ['chosen', 'rejected']:
                self.assertEqual(set(row[name]), {'from', 'value'})
                self.assertEqual(row[name]['from'], 'gpt')

    def test_registry_matches_public_filenames(self):
        registry = json.loads((ROOT / 'examples/dataset_info.json').read_text())
        self.assertEqual(set(registry), {'sage_sft', 'sage_dpo'})
        for info in registry.values():
            self.assertEqual(info['formatting'], 'sharegpt')
        self.assertEqual(registry['sage_sft']['file_name'], 'sft_train.json')
        self.assertEqual(registry['sage_dpo']['file_name'], 'preferences.json')
        self.assertTrue(registry['sage_dpo']['ranking'])

    def test_dpo_public_export_excludes_private_fields(self):
        tree = ast.parse((ROOT / 'dpo/dpo_core.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'public_preference_record')
        ns = {'Any': Any}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<export>', 'exec'), ns)
        sample = json.loads((FIXTURES / 'sharegpt_dpo.json').read_text())[0]
        internal = {**sample, 'sample_id': 'private', 'chosen_reward': 1.0, 'source': 'private'}
        self.assertEqual(ns['public_preference_record'](internal), sample)
        self.assertIn('sample_id', internal)

    @staticmethod
    def adapter():
        spec = importlib.util.spec_from_file_location('prepare_rl_data', ROOT / 'scripts/prepare_rl_data.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.convert_record

    def test_rl_adapter_derives_targets_without_metadata(self):
        records = json.loads((FIXTURES / 'sharegpt_sft.json').read_text())
        before = json.dumps(records)
        full = self.adapter()(records[3], 1)
        gt = json.loads(full['reward_ground_truth'])
        self.assertEqual(gt['answer'], 'Source answer.')
        self.assertEqual(gt['reference_boxes'], [[10, 20, 60, 80]])
        self.assertTrue(gt['full_LVKC'])
        self.assertEqual(full['prompt'], records[3]['conversations'][0]['value'])
        direct = json.loads(self.adapter()(records[4], 2)['reward_ground_truth'])
        self.assertEqual(direct['answer'], 'Source answer.')
        self.assertFalse(direct['full_LVKC'])
        self.assertEqual(json.dumps(records), before)

    def test_rl_adapter_rejects_partial_supervision(self):
        records = json.loads((FIXTURES / 'sharegpt_sft.json').read_text())
        for row in records[:3]:
            with self.assertRaisesRegex(ValueError, 'terminal answer'):
                self.adapter()(row, 1)

    def test_rl_adapter_rejects_malformed_boxes_and_image_mismatch(self):
        records = json.loads((FIXTURES / 'sharegpt_sft.json').read_text())
        row = records[3]
        row['conversations'][1]['value'] = row['conversations'][1]['value'].replace('[10,20,60,80]', '[60,20,10,80]')
        with self.assertRaisesRegex(ValueError, 'XYXY'):
            self.adapter()(row, 1)
        row = records[4]
        row['conversations'][0]['value'] = 'No image placeholder'
        with self.assertRaisesRegex(ValueError, 'placeholder'):
            self.adapter()(row, 1)

    def test_source_syntax(self):
        for path in ROOT.rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))

    def test_json_syntax(self):
        for path in ROOT.rglob('*.json'):
            json.loads(path.read_text(encoding='utf-8'))

    def test_full_reward_and_continuous_iou(self):
        good, partial = compute_score([
            {'response': response(), 'ground_truth': ground_truth()},
            {'response': response([0, 0, 5, 5]), 'ground_truth': ground_truth()},
        ])
        self.assertAlmostEqual(good['overall'], 1.0)
        self.assertAlmostEqual(partial['iou'], 0.25)
        self.assertAlmostEqual(partial['overall'], 0.775)

    def test_invalid_format_and_box_count_gate(self):
        invalid = response().replace('</VIS>', '')
        self.assertEqual(compute_score([{'response': invalid, 'ground_truth': ground_truth()}])[0]['overall'], 0.0)
        two_boxes = response().replace('[[0, 0, 10, 10]]', '[[0, 0, 10, 10], [20, 20, 30, 30]]')
        self.assertEqual(compute_score([{'response': two_boxes, 'ground_truth': ground_truth()}])[0]['overall'], 0.0)

    def test_single_stage_formats_and_direct_answer(self):
        for stage, tag in [('L', 'LOC'), ('V', 'VIS'), ('K', 'KNO'), ('C', 'CON')]:
            gt = json.dumps({'task_type': stage})
            self.assertEqual(format_reward(f'<{tag}>synthetic</{tag}>', gt), 1.0)
        self.assertEqual(format_reward('yes', json.dumps({'task_type': 'C'})), 1.0)

    def test_trainer_stage_extraction(self):
        tree = ast.parse((ROOT / 'easyr1/verl/trainer/ray_trainer.py').read_text())
        names = {'_sagrpo_extract_stage_text', '_sagrpo_stage_char_spans'}
        nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
        self.assertEqual(len(nodes), 2)
        for node in nodes:
            node.decorator_list = []
        namespace = {'re': re}
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), '<copied-parser>', 'exec'), namespace)
        text = response()
        for stage in ['L', 'V', 'K', 'C']:
            self.assertTrue(namespace['_sagrpo_extract_stage_text'](text, stage))
            spans = namespace['_sagrpo_stage_char_spans'](text)[stage]
            self.assertEqual(len(spans), 1)
            self.assertEqual(text[slice(*spans[0])], namespace['_sagrpo_extract_stage_text'](text, stage))


if __name__ == '__main__':
    unittest.main()
