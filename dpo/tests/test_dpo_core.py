import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dpo_core import (  # noqa: E402
    IGNORE_INDEX,
    is_forbidden_benchmark_test_row,
    response_mask,
    select_same_prompt_pair,
    sequence_logps,
    standard_dpo_loss,
)


def row(sample_id="s1"):
    return {
        "sample_id": sample_id,
        "source": "DeepLesion",
        "prompt": "<image> q",
        "images": ["x.png"],
        "reward_ground_truth": {"task_type": "C", "full_LVKC": False},
    }


def scorer(items):
    values = {"best": 0.8, "worst": 0.1, "tie": 0.8}
    return [
        {"overall": values[x["response"]], "format": 1.0, "explicit_conclusion": 1.0, "bbox_count_match": 1.0}
        for x in items
    ]


def test_1_chosen_reward_strictly_greater():
    pair = select_same_prompt_pair(row(), ["worst", "best"], scorer)
    assert pair is not None
    assert pair.chosen_score["overall"] > pair.rejected_score["overall"]
    assert select_same_prompt_pair(row(), ["best", "tie"], scorer) is None


def test_2_same_sample_only_and_benchmark_labels_forbidden():
    pair = select_same_prompt_pair(row("one"), ["worst", "best"], scorer)
    assert pair is not None
    forbidden = row("test") | {"source": "BENCHMARK_TEST_ORACLE", "benchmark": "VQA-RAD"}
    assert is_forbidden_benchmark_test_row(forbidden)
    with pytest.raises(ValueError):
        select_same_prompt_pair(forbidden, ["worst", "best"], scorer)


def test_3_reference_parameters_are_frozen():
    ref = torch.nn.Linear(3, 2)
    ref.requires_grad_(False)
    with torch.no_grad():
        out = ref(torch.ones(1, 3))
    assert not out.requires_grad
    assert all(not p.requires_grad and p.grad is None for p in ref.parameters())


def test_4_prompt_and_padding_are_not_in_response_logprob():
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 2, 3, IGNORE_INDEX]])
    assert response_mask(labels).tolist() == [[False, False, True, True, False]]
    logits = torch.zeros(1, 5, 5)
    base = sequence_logps(logits, labels)
    logits[:, 0, :] = 1000  # predicts the masked prompt position only
    logits[:, 3, :] = -1000  # predicts a padding position only
    assert torch.allclose(sequence_logps(logits, labels), base)


def test_5_loss_falls_when_policy_relative_preference_increases():
    ref_c = torch.tensor([0.0])
    ref_r = torch.tensor([0.0])
    weak = standard_dpo_loss(torch.tensor([0.1]), torch.tensor([0.0]), ref_c, ref_r, 0.1)
    strong = standard_dpo_loss(torch.tensor([2.0]), torch.tensor([0.0]), ref_c, ref_r, 0.1)
    assert strong.item() < weak.item()


def test_contradictory_answer_is_rejected_by_shared_reward_ranking():
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'rewards'))
    from combined_reward import compute_score
    sample = row()
    sample['reward_ground_truth'] = {'answer': 'B', 'full_LVKC': True,
                                     'reference_boxes': [[0, 0, 10, 10]]}
    template = ('<LOC>{"bbox_xyxy_pixel":[[0,0,10,10]]}</LOC>'
                '<VIS>Finding.</VIS><KNO>Relation.</KNO><CON>{}</CON>')
    good = template.replace('{}', 'B')
    bad = template.replace('{}', 'B. The correct option is C.')
    pair = select_same_prompt_pair(sample, [bad, good], compute_score)
    assert pair.chosen == good
    assert pair.rejected == bad
    assert pair.rejected_score['accuracy'] == 0.0

