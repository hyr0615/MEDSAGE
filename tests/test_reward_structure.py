import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rewards'))
from combined_reward import compute_score
from format_reward import format_reward
from iou_reward import parse_boxes_from_text, iou_reward_details


class RewardStructureTests(unittest.TestCase):
    def test_empty_and_nested_stage_content_fails_format(self):
        for response in ['<VIS> </VIS>', '<VIS>\n </VIS>', '<VIS><extra>finding</extra></VIS>']:
            self.assertEqual(format_reward(response, {'task_type': 'V'}), 0.0)
        self.assertEqual(format_reward('<extra>yes</extra>', {'task_type': 'C'}), 0.0)

    def test_public_and_legacy_stage_availability(self):
        for availability in [{'V': True}, {'VISUAL': True}, {'VIS': True}]:
            self.assertEqual(format_reward('<VIS>finding</VIS>', {'stage_availability': availability}), 1.0)

    def test_dict_and_json_reward_ground_truth_agree(self):
        gt = {'answer': 'yes', 'task_type': 'full_LVKC', 'reference_boxes': [[0, 0, 10, 10]]}
        text = '<LOC>{"bbox_xyxy_pixel":[[0,0,10,10]]}</LOC><VIS>finding</VIS><KNO>relation</KNO><CON>yes</CON>'
        scores = compute_score([{'response': text, 'ground_truth': value} for value in [gt, json.dumps(gt)]])
        self.assertEqual(scores[0], scores[1])
        self.assertAlmostEqual(scores[0]['overall'], 1.0)

    def test_invalid_box_cannot_be_hidden_next_to_valid_box(self):
        for bad in [[], [True, 0, 5, 5], [1, 1, 0, 0], [-1, 0, 5, 5], [0, 0, 5], [0, 0, float('nan'), 5]]:
            payload = json.dumps({'bbox_xyxy_pixel': [[0, 0, 10, 10], bad]})
            prediction = '<LOC>' + payload + '</LOC>'
            self.assertEqual(parse_boxes_from_text(prediction), [])
            self.assertEqual(iou_reward_details(prediction, {'reference_boxes': [[0, 0, 10, 10]]})['bbox_count_match'], 0.0)

    def test_explicit_xywh_is_not_silently_scored_as_xyxy(self):
        text = '<LOC>{"bbox_format":"xywh","bbox":[1,2,10,20]}</LOC>'
        self.assertEqual(parse_boxes_from_text(text), [])

    def test_supported_box_representations_remain_valid(self):
        for payload in [[1, 2, 10, 20], {'bbox': [1, 2, 10, 20]},
                        {'bbox_xyxy_pixel': [[1, 2, 10, 20]]},
                        {'x_min': 1, 'y_min': 2, 'x_max': 10, 'y_max': 20}]:
            self.assertEqual(parse_boxes_from_text('<LOC>' + json.dumps(payload) + '</LOC>'), [[1, 2, 10, 20]])

    def test_large_coordinates_cannot_produce_nan_reward(self):
        text = '<LOC>{"bbox_xyxy_pixel":[[0,0,1e308,1e308]]}</LOC>'
        result = iou_reward_details(text, {'reference_boxes': [[0, 0, 1e308, 1e308]]})
        self.assertEqual(result['iou'], 0.0)

