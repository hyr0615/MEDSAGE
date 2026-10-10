import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))

from accuracy_reward import accuracy_reward, extract_tag, parse_ground_truth
from format_reward import format_reward
from iou_reward import iou_reward_details
from stage_direct_reward import score_direct_k, score_direct_v
from stage_final_correctness import stage_c_correctness


def compute_score(
    reward_inputs: list[dict[str, Any]],
    accuracy_weight: float = 0.6,
    iou_weight: float = 0.3,
    format_weight: float = 0.1,
) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use reward_type=batch for MEDSAGE combined reward.")
    scores = []
    for item in reward_inputs:
        response = item.get("response", "")
        ground_truth = item.get("ground_truth", "")
        acc = accuracy_reward(response, ground_truth)
        iou_details = iou_reward_details(response, ground_truth)
        iou_score = iou_details["iou"]
        fmt = format_reward(response, ground_truth)
        # Structural validity gates the entire trajectory reward.
        weighted = accuracy_weight * acc + iou_weight * iou_score + format_weight * fmt
        overall = weighted if fmt == 1.0 else 0.0
        gt = parse_ground_truth(ground_truth)
        direct_spec = gt.get("stage_direct_spec") or {}
        visual_text = extract_tag(response, "VIS")
        knowledge_text = extract_tag(response, "KNO")
        v_direct = score_direct_v(visual_text, direct_spec.get("V"))
        k_direct = score_direct_k(knowledge_text, direct_spec.get("K"))
        conclusion_required = bool(gt.get("full_LVKC")) or gt.get("task_type") in {
            "full_LVKC",
            "C",
        }
        stage_availability = gt.get("stage_availability") or {}
        c_only = (
            bool(stage_availability.get("C"))
            and not any(stage_availability.get(stage) for stage in ("L", "V", "K"))
        )
        localization_required = (
            bool(gt.get("full_LVKC"))
            or gt.get("task_type") in {"full_LVKC", "L"}
            or bool(stage_availability.get("L"))
        )
        has_any_stage_tag = any(
            (f"<{tag}>" in response.upper() or f"</{tag}>" in response.upper())
            for tag in ("LOC", "VIS", "KNO", "CON")
        )
        direct_c = c_only and bool(response.strip()) and not has_any_stage_tag
        explicit_conclusion = 1.0 if extract_tag(response, "CON") or direct_c else 0.0
        if conclusion_required and explicit_conclusion == 0.0:
            overall = 0.0
        if localization_required and iou_details["bbox_count_match"] == 0.0:
            overall = 0.0
        scores.append(
            {
                "overall": overall,
                "accuracy": acc,
                "iou": iou_score,
                "best_iou": iou_details["best_iou"],
                "format": fmt,
                "explicit_conclusion": explicit_conclusion,
                "pred_box_count": iou_details["pred_box_count"],
                "reference_box_count": iou_details["reference_box_count"],
                "bbox_count_match": iou_details["bbox_count_match"],
                "stage_v_direct_eligible": float(v_direct is not None),
                "stage_k_direct_eligible": float(k_direct is not None),
                "stage_v_direct_score": float(v_direct or 0.0),
                "stage_k_direct_score": float(k_direct or 0.0),
                "stage_c_correctness": stage_c_correctness(response, ground_truth),
            }
        )
    return scores
