import re
from typing import Any


TAG_ORDER = ["LOC", "VIS", "KNO", "CON"]


try:
    from .accuracy_reward import parse_ground_truth
except ImportError:
    from accuracy_reward import parse_ground_truth


def has_tag(text: str, tag: str) -> bool:
    return re.search(rf"<{tag}>\s*.+?\s*</{tag}>", text or "", flags=re.I | re.S) is not None


def tag_counts(text: str, tag: str) -> tuple[int, int, int]:
    opening = len(re.findall(rf"<{tag}>", text or "", flags=re.I))
    closing = len(re.findall(rf"</{tag}>", text or "", flags=re.I))
    complete = len(re.findall(rf"<{tag}>\s*.+?\s*</{tag}>", text or "", flags=re.I | re.S))
    return opening, closing, complete


def tags_in_order(text: str, tags: list[str]) -> bool:
    pos = -1
    for tag in tags:
        m = re.search(rf"<{tag}>", text or "", flags=re.I)
        if not m or m.start() <= pos:
            return False
        pos = m.start()
    return True


def exact_tag_sequence(text: str, tags: list[str]) -> bool:
    sections = [rf"<{tag}>\s*.+?\s*</{tag}>" for tag in tags]
    pattern = r"^\s*" + r"\s*".join(sections) + r"\s*$"
    if re.fullmatch(pattern, text or "", flags=re.I | re.S) is None:
        return False
    for tag in tags:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.I | re.S)
        if not match.group(1).strip() or re.search(r"</?[A-Za-z][^>]*>", match.group(1)):
            return False
    return True


def expected_tags(gt: dict[str, Any]) -> list[str]:
    if gt.get("full_LVKC") or gt.get("task_type") == "full_LVKC":
        return TAG_ORDER
    task = gt.get("task_type")
    if task == "L":
        return ["LOC"]
    if task == "V":
        return ["VIS"]
    if task == "K":
        return ["KNO"]
    if task == "C":
        return ["CON"]
    stage = gt.get("stage_availability") or {}
    aliases = [('LOC', 'L'), ('VIS', 'V', 'VISUAL'), ('KNO', 'K', 'KNOWLEDGE'), ('CON', 'C', 'CONCLUSION')]
    return [tag for tag, keys in zip(TAG_ORDER, aliases) if any(stage.get(key) for key in keys)]


def format_reward(response: str, ground_truth: str) -> float:
    gt = parse_ground_truth(ground_truth)
    required = expected_tags(gt)
    if not required:
        return 0.0
    activity = {tag: tag_counts(response, tag) for tag in TAG_ORDER}
    # C-only targets may use untagged answers.
    if required == ["CON"] and not any(any(counts) for counts in activity.values()):
        return float(bool((response or "").strip()) and re.search(r"</?[A-Za-z][^>]*>", response) is None)
    # Extra stages must still be complete, unique and ordered.
    present = [tag for tag in TAG_ORDER if any(activity[tag])]
    if not set(required).issubset(present):
        return 0.0
    exact_once = all(tag_counts(response, tag) == (1, 1, 1) for tag in present)
    ordered = len(present) == 1 or tags_in_order(response, present)
    exact_structure = exact_tag_sequence(response, present)
    return 1.0 if exact_once and ordered and exact_structure else 0.0


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use reward_type=batch for MEDSAGE format reward.")
    return [{"overall": format_reward(x.get("response", ""), x.get("ground_truth", "")), "format": format_reward(x.get("response", ""), x.get("ground_truth", ""))} for x in reward_inputs]
