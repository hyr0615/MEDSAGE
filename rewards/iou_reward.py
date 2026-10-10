import json
import math
import re
from functools import lru_cache
from typing import Any


try:
    from .accuracy_reward import parse_ground_truth
except ImportError:
    from accuracy_reward import parse_ground_truth


def extract_l_payload(text: str) -> str:
    match = re.search(r"<LOC>\s*(.*?)\s*</LOC>", text or "", flags=re.I | re.S)
    return match.group(1).strip() if match else ""


def parse_boxes_from_text(text: str) -> list[list[float]]:
    payload = extract_l_payload(text)
    if not payload:
        return []
    try:
        data = json.loads(payload)
    except Exception:
        return []
    if isinstance(data, dict) and data.get("bbox_format", "xyxy_pixel") != "xyxy_pixel":
        return []

    def iter_boxes(value: Any):
        if isinstance(value, dict):
            for key in ("bbox_xyxy_pixel", "box", "bbox"):
                if key in value:
                    yield from iter_boxes(value[key])
                    return
            keys = ("x_min", "y_min", "x_max", "y_max")
            if all(key in value for key in keys):
                yield [value[key] for key in keys]
                return
            raise ValueError("Unrecognized box object.")
        if isinstance(value, list):
            if not value:
                raise ValueError("Empty box entry.")
            if len(value) == 4 and all(type(item) in (int, float) for item in value):
                yield value
                return
            for item in value:
                yield from iter_boxes(item)
            return
        raise ValueError("Malformed box coordinates.")

    boxes = []
    try:
        for box in iter_boxes(data):
            if any(type(v) not in (int, float) for v in box):
                return []
            x1, y1, x2, y2 = [float(v) for v in box]
            if not (all(math.isfinite(v) and v >= 0 for v in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1):
                return []
            boxes.append([x1, y1, x2, y2])
    except (TypeError, ValueError):
        return []
    return boxes


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    if not all(math.isfinite(value) for value in (inter, area_a, area_b, area_a + area_b - inter)):
        return 0.0
    return inter / max(area_a + area_b - inter, 1e-6)


def _coerce_boxes(value: Any) -> list[list[float]]:
    boxes: list[list[float]] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key in ("bbox_xyxy_pixel", "box", "bbox"):
                if key in item:
                    visit(item[key])
                    return
            return
        if isinstance(item, (list, tuple)):
            if len(item) == 4 and all(isinstance(v, (int, float)) for v in item):
                x1, y1, x2, y2 = [float(v) for v in item]
                if all(math.isfinite(v) for v in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                return
            for child in item:
                visit(child)

    visit(value)
    return boxes


def _one_to_one_ious(
    predicted: list[list[float]], reference: list[list[float]]
) -> list[float]:
    if max(len(predicted), len(reference)) <= 12:
        smaller, larger = (
            (predicted, reference)
            if len(predicted) <= len(reference)
            else (reference, predicted)
        )
        matrix = [[iou(left, right) for right in larger] for left in smaller]

        @lru_cache(maxsize=None)
        def solve(row: int, used_mask: int) -> tuple[float, tuple[float, ...]]:
            if row == len(smaller):
                return 0.0, ()
            best_total = -1.0
            best_values: tuple[float, ...] = ()
            for column, value in enumerate(matrix[row]):
                if used_mask & (1 << column):
                    continue
                tail_total, tail_values = solve(row + 1, used_mask | (1 << column))
                total = value + tail_total
                if total > best_total:
                    best_total = total
                    best_values = (value, *tail_values)
            return best_total, best_values

        return list(solve(0, 0)[1])

    # Use greedy matching above 12 boxes to bound assignment cost.
    candidates = sorted(
        (
            (iou(pred_box, ref_box), pred_idx, ref_idx)
            for pred_idx, pred_box in enumerate(predicted)
            for ref_idx, ref_box in enumerate(reference)
        ),
        reverse=True,
    )
    used_pred: set[int] = set()
    used_ref: set[int] = set()
    matches: list[float] = []
    for value, pred_idx, ref_idx in candidates:
        if pred_idx in used_pred or ref_idx in used_ref:
            continue
        used_pred.add(pred_idx)
        used_ref.add(ref_idx)
        matches.append(value)
        if len(matches) == min(len(predicted), len(reference)):
            break
    return matches


def iou_reward_details(response: str, ground_truth: str) -> dict[str, float]:
    gt = parse_ground_truth(ground_truth)
    ref_boxes = _coerce_boxes(gt.get("reference_boxes"))
    if not ref_boxes:
        answer = gt.get("answer", "")
        ref_boxes = parse_boxes_from_text(answer) if isinstance(answer, str) else []
    pred_boxes = parse_boxes_from_text(response)
    if not ref_boxes or not pred_boxes:
        return {
            "iou": 0.0,
            "best_iou": 0.0,
            "pred_box_count": float(len(pred_boxes)),
            "reference_box_count": float(len(ref_boxes)),
            "bbox_count_match": 0.0,
        }

    matched = _one_to_one_ious(pred_boxes, ref_boxes)
    best = max(matched, default=0.0)
    # Unmatched boxes contribute zero to the count-normalized mean.
    matched_iou = sum(matched) / max(len(pred_boxes), len(ref_boxes))
    return {
        "iou": matched_iou,
        "best_iou": best,
        "pred_box_count": float(len(pred_boxes)),
        "reference_box_count": float(len(ref_boxes)),
        "bbox_count_match": 1.0 if len(pred_boxes) == len(ref_boxes) else 0.0,
    }


def iou_reward(response: str, ground_truth: str, threshold: float | None = None) -> tuple[float, float]:
    """Return count-aware continuous IoU; threshold is API-compatible only."""
    details = iou_reward_details(response, ground_truth)
    return details["iou"], details["best_iou"]


def compute_score(reward_inputs: list[dict[str, Any]], threshold: float = 0.5) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use reward_type=batch for MEDSAGE IoU reward.")
    scores = []
    for x in reward_inputs:
        details = iou_reward_details(x.get("response", ""), x.get("ground_truth", ""))
        scores.append({"overall": details["iou"], **details})
    return scores
